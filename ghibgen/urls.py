# ghibgen/urls.py
from django.urls import path
from . import views, api_views

app_name = "ghibgen"

urlpatterns = [
    # Original template-based views (for backward compatibility)
    path("", views.index, name="index"),
    path("generate/", views.index, name="generate"),
    
    # REST API endpoints for React frontend
    path("api/generate/", api_views.generate_image_api, name="api_generate"),
    path("api/config/", api_views.get_config_api, name="api_config"),
]
