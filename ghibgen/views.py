# ghibgen/views.py
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

from .forms import GenerateForm
from .pipeline_sd import generate, generate_img2img
from .utils import ensure_media, unique_png


def _apply_text2img_preset(preset: str, guidance, steps):
    """Return (guidance, steps) if values are missing; otherwise keep user overrides."""
    if guidance is None and steps is None:
        if preset == "speed":
            return 7.0, 14
        if preset == "quality":
            return 8.0, 24
        # balanced default
        return 7.5, 18
    # partial override support
    if guidance is None:
        guidance = {"speed":7.0, "quality":8.0}.get(preset, 7.5)
    if steps is None:
        steps = {"speed":14, "quality":24}.get(preset, 18)
    return guidance, steps


def _apply_img2img_preset(preset: str, guidance, steps, strength):
    """Return (guidance, steps, strength) if values are missing; otherwise keep user overrides."""
    # defaults tuned for realism and speed (matching pipeline defaults)
    table = {
        "faithful":  (6.0, 14, 0.25),
        "balanced":  (6.5, 16, 0.35),
        "stylized":  (7.0, 18, 0.55),
        "speed":     (6.3, 12, 0.30),  # useful if someone selects Speed with img2img
        "quality":   (7.2, 20, 0.40),
    }
    g_def, s_def, str_def = table.get(preset, table["balanced"])
    return (
        g_def if guidance is None else guidance,
        s_def if steps    is None else steps,
        str_def if strength is None else strength,
    )


@csrf_exempt
def index(request):
    form = GenerateForm(request.POST or None, request.FILES or None)
    img_url, error = None, None

    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        try:
            ensure_media()

            # Common
            preset   = data.get("preset") or "balanced"
            prompt   = data["prompt"]
            negative = data.get("negative_prompt") or ""
            aspect   = data.get("aspect") or "3:2"
            seed     = data.get("seed") or None
            lora     = data.get("lora") or None
            upscale_mode   = data.get("upscale_mode") or "auto"
            upscale_factor = int(data.get("upscale_factor") or "2")

            # Numeric inputs (may be None if user leaves blank)
            guidance = data.get("guidance_scale")
            steps    = data.get("num_inference_steps")

            # Img2img?
            init = data.get("init_image")
            if init:
                strength = data.get("strength")
                guidance, steps, strength = _apply_img2img_preset(preset, guidance, steps, strength)
                pil_init = Image.open(init)
                pil_img = generate_img2img(
                    prompt=prompt,
                    init_image=pil_init,
                    negative=negative,
                    guidance=guidance,
                    steps=steps,
                    strength=float(strength),
                    seed=seed,
                    lora=lora,
                    speed_mode=True,
                    upscale_mode=upscale_mode,
                    upscale_factor=upscale_factor,
                )
            else:
                guidance, steps = _apply_text2img_preset(preset, guidance, steps)
                pil_img = generate(
                    prompt=prompt,
                    negative=negative,
                    guidance=guidance,
                    steps=steps,
                    aspect=aspect,
                    seed=seed,
                    lora=lora,
                    speed_mode=True,
                    upscale_mode=upscale_mode,
                    upscale_factor=upscale_factor,
                )

            # Save and expose URL
            fname = unique_png()
            out_path = settings.MEDIA_ROOT / fname
            pil_img.save(out_path)
            img_url = settings.MEDIA_URL + fname

        except Exception as e:
            error = f"Generation failed: {e}"

    return render(request, "index.html", {"form": form, "img_url": img_url, "error": error})
