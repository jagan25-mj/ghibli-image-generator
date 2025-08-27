from django.urls import path
from . import views

app_name = "ghibgen"

urlpatterns = [
    path("", views.index, name="index"),
    path("generate/", views.index, name="generate"),  # same view handles POST
]
