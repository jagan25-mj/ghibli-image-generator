# ghibgen/api_views.py
"""
REST API views for the Ghibli Image Generator.
Provides JSON endpoints for the React frontend.
"""
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image
import json
import logging
import random

from .pipeline_sd import generate, generate_img2img
from .utils import ensure_media, unique_png

logger = logging.getLogger(__name__)


def _apply_text2img_preset(preset: str, guidance, steps):
    """Apply preset configuration for text-to-image generation."""
    if guidance is None and steps is None:
        if preset == "speed":
            return 7.0, 14
        if preset == "quality":
            return 8.0, 24
        return 7.5, 18
    if guidance is None:
        guidance = {"speed": 7.0, "quality": 8.0}.get(preset, 7.5)
    if steps is None:
        steps = {"speed": 14, "quality": 24}.get(preset, 18)
    return guidance, steps


def _apply_img2img_preset(preset: str, guidance, steps, strength):
    """Apply preset configuration for image-to-image generation."""
    table = {
        "faithful": (6.0, 14, 0.25),
        "balanced": (6.5, 16, 0.35),
        "stylized": (7.0, 18, 0.55),
        "speed": (6.3, 12, 0.30),
        "quality": (7.2, 20, 0.40),
    }
    g_def, s_def, str_def = table.get(preset, table["balanced"])
    return (
        g_def if guidance is None else guidance,
        s_def if steps is None else steps,
        str_def if strength is None else strength,
    )


@csrf_exempt
@require_http_methods(["POST"])
def generate_image_api(request):
    """
    API endpoint for image generation.
    Accepts JSON payload with generation parameters.
    Returns JSON with image URL and metadata.
    """
    try:
        # Parse request data
        if request.content_type == "application/json":
            data = json.loads(request.body)
        else:
            # Handle form data (multipart for image uploads)
            data = {
                "preset": request.POST.get("preset", "balanced"),
                "prompt": request.POST.get("prompt", ""),
                "negative_prompt": request.POST.get("negative_prompt", ""),
                "aspect": request.POST.get("aspect", "3:2"),
                "lora": request.POST.get("lora"),
                "upscale_mode": request.POST.get("upscale_mode", "auto"),
                "upscale_factor": request.POST.get("upscale_factor", "2"),
                "guidance_scale": request.POST.get("guidance_scale"),
                "num_inference_steps": request.POST.get("num_inference_steps"),
                "seed": request.POST.get("seed"),
                "strength": request.POST.get("strength"),
            }
            if "init_image" in request.FILES:
                data["init_image"] = request.FILES["init_image"]

        # Validate required fields
        if not data.get("prompt"):
            return JsonResponse(
                {"error": "Prompt is required"}, status=400
            )

        # Extract and process parameters
        ensure_media()
        
        preset = data.get("preset") or "balanced"
        prompt = data["prompt"]
        negative = data.get("negative_prompt") or ""
        aspect = data.get("aspect") or "3:2"
        lora = data.get("lora") or None
        upscale_mode = data.get("upscale_mode") or "auto"
        upscale_factor = int(data.get("upscale_factor") or "2")

        # Parse numeric parameters
        guidance = None
        steps = None
        if data.get("guidance_scale"):
            try:
                guidance = float(data["guidance_scale"])
            except (ValueError, TypeError):
                pass
        if data.get("num_inference_steps"):
            try:
                steps = int(data["num_inference_steps"])
            except (ValueError, TypeError):
                pass

        # Handle seed
        seed = data.get("seed")
        if seed:
            try:
                seed = int(seed)
            except (ValueError, TypeError):
                seed = None
        if seed is None:
            seed = random.randrange(0, 2**31 - 1)

        # Process init image if provided
        init_file = data.get("init_image")
        if init_file:
            strength = data.get("strength")
            if strength:
                try:
                    strength = float(strength)
                except (ValueError, TypeError):
                    strength = None
            
            guidance, steps, strength = _apply_img2img_preset(
                preset, guidance, steps, strength
            )
            
            with Image.open(init_file) as im:
                pil_init = im.convert("RGB")
            
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
            mode = "img2img"
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
            mode = "text2img"

        # Save generated image
        fname = unique_png()
        out_path = settings.MEDIA_ROOT / fname
        pil_img.save(out_path)
        img_url = settings.MEDIA_URL + fname

        # Prepare response
        response_data = {
            "success": True,
            "image_url": img_url,
            "metadata": {
                "preset": preset,
                "guidance": guidance,
                "steps": steps,
                "aspect": aspect,
                "seed": seed,
                "upscale_mode": upscale_mode,
                "upscale_factor": upscale_factor,
                "lora": lora or "—",
                "mode": mode,
            },
        }

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("API generation failed")
        return JsonResponse(
            {"success": False, "error": str(e)}, status=500
        )


@require_http_methods(["GET"])
def get_config_api(request):
    """
    API endpoint to get available configuration options.
    Returns presets, aspects, upscale modes, etc.
    """
    config = {
        "presets": [
            {"value": "balanced", "label": "Balanced"},
            {"value": "speed", "label": "Speed"},
            {"value": "quality", "label": "Quality"},
            {"value": "faithful", "label": "Faithful (img→img)"},
            {"value": "stylized", "label": "Stylized (img→img)"},
        ],
        "aspects": [
            {"value": "1:1", "label": "Square (1:1)"},
            {"value": "3:2", "label": "Landscape (3:2)"},
            {"value": "16:9", "label": "Wide (16:9)"},
            {"value": "4:5", "label": "Portrait (4:5)"},
        ],
        "upscale_modes": [
            {"value": "auto", "label": "Auto (Real-ESRGAN → Lanczos)"},
            {"value": "realesrgan", "label": "Real-ESRGAN"},
            {"value": "lanczos", "label": "Lanczos (fast)"},
            {"value": "off", "label": "Off"},
        ],
        "upscale_factors": [
            {"value": "2", "label": "2×"},
            {"value": "4", "label": "4×"},
        ],
        "defaults": {
            "preset": "balanced",
            "aspect": "3:2",
            "guidance_scale": 7.5,
            "num_inference_steps": 18,
            "strength": 0.6,
            "upscale_mode": "auto",
            "upscale_factor": "2",
        },
    }
    return JsonResponse(config)