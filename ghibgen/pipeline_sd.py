# ghibgen/pipeline_sd.py
from __future__ import annotations

from typing import Optional, Tuple, Dict
import os, shutil, subprocess, tempfile
import torch
from PIL import Image

# ---------- Global perf knobs ----------
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    except Exception:
        pass

# ---------- Env / compat ----------
os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from transformers import AddedToken
except Exception:  # pragma: no cover
    from transformers.tokenization_utils_base import AddedToken  # type: ignore

if not getattr(AddedToken, "__hash__", None) or AddedToken.__hash__ is object.__hash__:
    def _addedtoken_hash(self: "AddedToken"):
        return hash((self.content, self.lstrip, self.rstrip, self.single_word, self.normalized))
    AddedToken.__hash__ = _addedtoken_hash  # type: ignore

if not hasattr(AddedToken, "__getstate__"):
    def _addedtoken_getstate(self):
        return {
            "content": getattr(self, "content", ""),
            "lstrip": getattr(self, "lstrip", False),
            "rstrip": getattr(self, "rstrip", False),
            "single_word": getattr(self, "single_word", False),
            "normalized": getattr(self, "normalized", True),
            "special": getattr(self, "special", False),
        }
    AddedToken.__getstate__ = _addedtoken_getstate  # type: ignore

if not hasattr(AddedToken, "__setstate__"):
    def _addedtoken_setstate(self, state):
        for k, v in (state or {}).items():
            try:
                setattr(self, k, v)
            except Exception:
                pass
    AddedToken.__setstate__ = _addedtoken_setstate  # type: ignore

# ---------- Diffusers ----------
try:
    from diffusers import (
        StableDiffusionPipeline,             # type: ignore
        StableDiffusionImg2ImgPipeline,      # type: ignore
        EulerAncestralDiscreteScheduler,     # type: ignore
    )
except Exception as e:
    raise ImportError("pip install diffusers transformers accelerate safetensors pillow") from e

try:
    from diffusers import DPMSolverMultistepScheduler  # type: ignore
    HAVE_DPM = True
except Exception:  # pragma: no cover
    DPMSolverMultistepScheduler = None  # type: ignore
    HAVE_DPM = False

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PIPE: Optional[StableDiffusionPipeline] = None
_PIPE_IMG2IMG: Optional[StableDiffusionImg2ImgPipeline] = None
_CUR_LORA: Optional[str] = None
_IP_ADAPTER_READY: bool = False

# ---------- Models ----------
BASE_MODEL = os.getenv("GHIBGEN_BASE_MODEL", "Lykon/dreamshaper-8")

# Allow overriding Real-ESRGAN path via env, but keep your Windows default as fallback.
REALESRGAN_EXE = os.getenv(
    "GHIBGEN_REALESRGAN_EXE",
    r"C:\tools\realesrgan-ncnn-vulkan-20220424\realesrgan-ncnn-vulkan.exe",
)

# ---------- Perf helpers ----------
def _maybe_enable_speed_tricks(pipe: StableDiffusionPipeline | StableDiffusionImg2ImgPipeline) -> None:
    if _DEVICE == "cuda":
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    # Memory offload helps low-VRAM GPUs; costs small overhead on high VRAM.
    try:
        if _DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
    except Exception:
        pass

def _pick_scheduler(pipe: StableDiffusionPipeline | StableDiffusionImg2ImgPipeline, speed_mode: bool) -> None:
    if speed_mode:
        try:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    if HAVE_DPM and DPMSolverMultistepScheduler is not None:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # type: ignore[attr-defined]
        except Exception:
            pass

# ---------- Real-ESRGAN helpers ----------
def _has_realesrgan() -> bool:
    if os.path.isfile(str(REALESRGAN_EXE)):
        return True
    return shutil.which("realesrgan-ncnn-vulkan") is not None

def upscale_lanczos(img: Image.Image, scale: int = 2) -> Image.Image:
    scale = max(1, int(scale))
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

def upscale_realesrgan(img: Image.Image, scale: int = 2) -> Image.Image:
    scale = max(1, int(scale))
    if not _has_realesrgan():
        return upscale_lanczos(img, scale=scale)
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.png")
        out = os.path.join(td, "out.png")
        img.save(inp, "PNG")
        exe = REALESRGAN_EXE if os.path.isfile(str(REALESRGAN_EXE)) else "realesrgan-ncnn-vulkan"
        cmd = [str(exe), "-i", inp, "-o", out, "-s", str(scale), "-f", "png", "-n", "realesr-animevideov3"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.isfile(out):
                with Image.open(out) as im:
                    return im.convert("RGB").copy()
        except Exception:
            pass
    return upscale_lanczos(img, scale=scale)

# ---------- Pipeline loaders ----------
def _load_pipe(speed_mode: bool = True) -> StableDiffusionPipeline:
    """
    Build (and memoize) SD1.5 text-to-image.
    Safety can be disabled for local dev with GHIBGEN_DISABLE_SAFETY=1
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

    _PIPE = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    )

    # Optional dev-only safety disable (do NOT disable for public apps)
    if os.getenv("GHIBGEN_DISABLE_SAFETY", "0") == "1":
        try:
            _PIPE.safety_checker = None
            if hasattr(_PIPE, "requires_safety_checker"):
                _PIPE.requires_safety_checker = False  # type: ignore[attr-defined]
        except Exception:
            pass

    _PIPE = _PIPE.to(_DEVICE)
    _PIPE.set_progress_bar_config(disable=True)
    _pick_scheduler(_PIPE, speed_mode)
    _maybe_enable_speed_tricks(_PIPE)
    return _PIPE

def _load_pipe_img2img(speed_mode: bool = True) -> StableDiffusionImg2ImgPipeline:
    """Share components with text2img to save RAM + downloads."""
    global _PIPE_IMG2IMG
    if _PIPE_IMG2IMG is not None:
        return _PIPE_IMG2IMG

    base = _load_pipe(speed_mode=speed_mode)
    _PIPE_IMG2IMG = StableDiffusionImg2ImgPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        unet=base.unet,
        scheduler=base.scheduler.__class__.from_config(base.scheduler.config),
        safety_checker=base.safety_checker,
        feature_extractor=base.feature_extractor,  # CLIPImageProcessor # type: ignore[attr-defined]
    ).to(base.device)
    _PIPE_IMG2IMG.set_progress_bar_config(disable=True)
    _pick_scheduler(_PIPE_IMG2IMG, speed_mode)
    _maybe_enable_speed_tricks(_PIPE_IMG2IMG)
    return _PIPE_IMG2IMG

# ---------- IP-Adapter (identity guidance) ----------
def _ensure_ip_adapter(pipe: StableDiffusionImg2ImgPipeline) -> None:
    global _IP_ADAPTER_READY
    if _IP_ADAPTER_READY:
        return
    candidates = [
        ("h94/IP-Adapter", "models", "ip-adapter-plus_sd15.safetensors"),
        ("h94/IP-Adapter", "models", "ip-adapter_sd15.safetensors"),
        ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin"),
    ]
    for repo, sub, weight in candidates:
        try:
            pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight)
            _IP_ADAPTER_READY = True
            return
        except Exception:
            pass

# ---------- Utils ----------
def _prepare_init_image(img: Image.Image, max_side: int = 768) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    w, h = img.size
    w = (w // 8) * 8 or 512
    h = (h // 8) * 8 or 512
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img

# ---------- Public APIs ----------
def set_lora(lora: Optional[str], speed_mode: bool = True) -> StableDiffusionPipeline:
    global _CUR_LORA
    pipe = _load_pipe(speed_mode=speed_mode)
    if not lora:
        return pipe
    if lora != _CUR_LORA:
        try:
            pipe.unload_lora_weights()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            pipe.load_lora_weights(lora)  # type: ignore[attr-defined]
            _CUR_LORA = lora
        except Exception:
            _CUR_LORA = None
    return pipe

def generate(
    prompt: str,
    negative: str,
    guidance: Optional[float],
    steps: int,
    aspect: str,
    seed: Optional[int],
    lora: Optional[str],
    speed_mode: bool = True,
    upscale_mode: str = "auto",   # "auto" | "realesrgan" | "lanczos" | "off"
    upscale_factor: int = 2,
) -> Image.Image:
    base_sizes: Dict[str, Tuple[int, int]] = {
        "1:1":  (512, 512),
        "3:2":  (768, 512),
        "16:9": (896, 504),
        "4:5":  (512, 640),
    }
    width, height = base_sizes.get(aspect, (512, 512))

    pipe = set_lora(lora, speed_mode=speed_mode)
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed)) if seed is not None else None

    guidance = 7.0 if guidance is None else float(guidance)
    steps = int(steps or 18)
    if speed_mode and steps > 28:
        steps = 24

    full_prompt = f"{prompt}, whimsical, painterly, soft color palette, gentle lighting, film grain"
    neg = f"{negative or ''}, watermark, text, low quality, deformed, nsfw"

    autocast_device = "cuda" if _DEVICE == "cuda" else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, enabled=_DEVICE == "cuda"):
        out = pipe(
            prompt=full_prompt,
            negative_prompt=neg,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
        )
    img = out.images[0]  # type: ignore[index]

    upscale_factor = max(1, int(upscale_factor))
    if upscale_mode == "off" or upscale_factor == 1:
        return img
    if upscale_mode == "realesrgan":
        return upscale_realesrgan(img, scale=upscale_factor)
    if upscale_mode == "lanczos":
        return upscale_lanczos(img, scale=upscale_factor)
    return upscale_realesrgan(img, scale=upscale_factor) if _has_realesrgan() else upscale_lanczos(img, scale=upscale_factor)

def generate_img2img(
    prompt: str,
    init_image: Image.Image,
    negative: str = "",
    guidance: Optional[float] = None,
    steps: int = 18,
    strength: float = 0.6,
    seed: Optional[int] = None,
    lora: Optional[str] = None,
    speed_mode: bool = True,
    upscale_mode: str = "auto",
    upscale_factor: int = 2,
    use_ip_adapter: bool = True,
    ip_scale: float = 0.6,
) -> Image.Image:
    pipe = set_lora(lora, speed_mode=speed_mode)
    i2i = _load_pipe_img2img(speed_mode=speed_mode)

    prep = _prepare_init_image(init_image, max_side=768)

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed)) if seed is not None else None
    guidance = 7.0 if guidance is None else float(guidance)
    steps = int(steps or 18)
    if speed_mode and steps > 28:
        steps = 24

    # Slightly sharper defaults improve fidelity for img2img without big cost:
    full_prompt = f"{prompt}, whimsical, painterly, soft color palette, gentle lighting, film grain, detailed textures"
    neg = f"{negative or ''}, watermark, text, low quality, deformed, nsfw"

    strength = float(max(0.05, min(0.95, strength)))

    kwargs = dict(
        prompt=full_prompt,
        negative_prompt=neg,
        image=prep,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    )

    if use_ip_adapter:
        _ensure_ip_adapter(i2i)
        if getattr(i2i, "ip_adapter", None):
            try:
                i2i.set_ip_adapter_scale(float(max(0.0, min(1.0, ip_scale))))
            except Exception:
                pass
            kwargs["ip_adapter_image"] = prep

    autocast_device = "cuda" if _DEVICE == "cuda" else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, enabled=_DEVICE == "cuda"):
        out = i2i(**kwargs)  # type: ignore

    img = out.images[0]  # type: ignore[index]

    upscale_factor = max(1, int(upscale_factor))
    if upscale_mode == "off" or upscale_factor == 1:
        return img
    if upscale_mode == "realesrgan":
        return upscale_realesrgan(img, scale=upscale_factor)
    if upscale_mode == "lanczos":
        return upscale_lanczos(img, scale=upscale_factor)
    return upscale_realesrgan(img, scale=upscale_factor) if _has_realesrgan() else upscale_lanczos(img, scale=upscale_factor)
