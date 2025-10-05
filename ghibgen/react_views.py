# ghibgen/react_views.py
"""
Views for serving the React frontend from Django (optional).
This is useful for integrated deployment where both frontend and backend are served from one server.
"""
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from pathlib import Path
import mimetypes


@require_http_methods(["GET"])
def serve_react_app(request, path=""):
    """
    Serve React build files from Django.
    This view is optional and only needed for integrated deployment.
    """
    react_build_dir = settings.BASE_DIR / "frontend" / "dist"
    
    # If path is empty, serve index.html
    if not path or path == "":
        path = "index.html"
    
    file_path = react_build_dir / path
    
    # Security check: ensure file is within build directory
    try:
        file_path = file_path.resolve()
        react_build_dir = react_build_dir.resolve()
        if not str(file_path).startswith(str(react_build_dir)):
            return HttpResponse("Invalid path", status=400)
    except Exception:
        return HttpResponse("File not found", status=404)
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        # For SPA routing, serve index.html for any non-existent routes
        index_path = react_build_dir / "index.html"
        if index_path.exists():
            with open(index_path, "rb") as f:
                return HttpResponse(f.read(), content_type="text/html")
        return HttpResponse("File not found", status=404)
    
    # Determine content type
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"
    
    # Serve the file
    with open(file_path, "rb") as f:
        return HttpResponse(f.read(), content_type=content_type)