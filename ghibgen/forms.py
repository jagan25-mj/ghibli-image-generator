# ghibgen/forms.py
from __future__ import annotations
from django import forms
from django.core.exceptions import ValidationError

ASPECT_CHOICES = [
    ("1:1",  "Square (1:1)"),
    ("3:2",  "Landscape (3:2)"),
    ("16:9", "Widescreen (16:9)"),
    ("4:5",  "Portrait (4:5)"),
]

UPSCALE_CHOICES = [
    ("auto",      "Auto (Real-ESRGAN → Lanczos)"),
    ("realesrgan","Real-ESRGAN"),
    ("lanczos",   "Lanczos"),
    ("off",       "Off"),
]

class GenerateForm(forms.Form):
    # Text-to-image prompt
    prompt = forms.CharField(
        label="Prompt",
        widget=forms.Textarea(
            attrs={
                "class": "field",
                "rows": 3,
                "placeholder": "A tranquil valley at sunrise, whimsical rooftops, soft painterly style",
            }
        ),
    )

    # Optional init image (img2img)
    init_image = forms.ImageField(
        label="Init image (optional)",
        required=False,
        help_text="PNG or JPEG. If provided, img2img is used."
    )

    # Prompt controls
    negative_prompt = forms.CharField(
        label="Negative prompt",
        required=False,
        widget=forms.TextInput(attrs={"class": "field", "placeholder": "low quality, text, watermark"}),
    )
    lora = forms.CharField(
        label="LoRA (optional path or repo id)",
        required=False,
        widget=forms.TextInput(attrs={"class": "field", "placeholder": "(optional)"}),
    )

    # Sampling / guidance
    guidance_scale = forms.FloatField(
        label="Guidance",
        initial=7.5,
        min_value=1.0,
        max_value=20.0,
        widget=forms.NumberInput(attrs={"class": "field", "step": "0.1"}),
    )
    num_inference_steps = forms.IntegerField(
        label="Steps",
        initial=18,
        min_value=6,
        max_value=60,
        widget=forms.NumberInput(attrs={"class": "field"}),
    )
    aspect = forms.ChoiceField(
        label="Aspect",
        choices=ASPECT_CHOICES,
        initial="3:2",
        widget=forms.Select(attrs={"class": "field select"}),
    )

    # Seed (optional integer)
    seed = forms.CharField(
        label="Seed",
        required=False,
        widget=forms.TextInput(attrs={"class": "field", "placeholder": "(optional)"}),
        help_text="Leave blank for random",
    )

    # Img2img-only strength
    strength = forms.FloatField(
        label="Strength (img2img)",
        required=False,
        initial=0.6,
        min_value=0.05,
        max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "field", "step": "0.05"}),
        help_text="How strongly to follow the prompt over the init image.",
    )

    # Upscale settings
    upscale_mode = forms.ChoiceField(
        label="Upscale mode",
        choices=UPSCALE_CHOICES,
        initial="auto",
        widget=forms.Select(attrs={"class": "field select"}),
    )
    upscale_factor = forms.ChoiceField(
        label="Upscale factor",
        choices=[("2", "2×"), ("4", "4×")],
        initial="2",
        widget=forms.Select(attrs={"class": "field select"}),
    )

    # ---- Cleaners / validation ----
    def clean_seed(self):
        seed = self.cleaned_data.get("seed", "").strip()
        if seed == "":
            return None
        try:
            return int(seed)
        except ValueError:
            raise ValidationError("Seed must be an integer or left blank.")

    def clean_init_image(self):
        img = self.cleaned_data.get("init_image")
        if not img:
            return img
        # Simple content-type & size checks
        if img.content_type not in ("image/png", "image/jpeg", "image/jpg"):
            raise ValidationError("Please upload a PNG or JPEG image.")
        # ~10MB cap; adjust if you like
        if img.size and img.size > 10 * 1024 * 1024:
            raise ValidationError("Image is too large (max 10 MB).")
        return img
