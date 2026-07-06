"""
File storage abstraction for Local ↔ Blob storage

This module provides a unified interface for file operations that works
with both local storage (for development) and Vercel Blob (for production).
"""

import os
import uuid
from werkzeug.utils import secure_filename

# Check if we're using Vercel Blob (production) or local storage (development)
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")
HAS_VERCEL_BLOB = False

if BLOB_READ_WRITE_TOKEN:
    try:
        from vercel_blob import put
        from vercel_blob import delete as blob_del
        from vercel_blob import list as blob_list
        HAS_VERCEL_BLOB = True
    except ImportError:
        print("Warning: vercel-blob package not installed, falling back to local file storage")
        HAS_VERCEL_BLOB = False

if HAS_VERCEL_BLOB:
    # Production: Use Vercel Blob
    print("Using Vercel Blob storage")
    
    def upload_file(file, upload_dir="uploads"):
        """Upload a file to Vercel Blob storage"""
        if not file or not file.filename:
            return None
            
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        # Read file content
        file_content = file.read()
        
        # Upload to Vercel Blob
        result = put(
            file_content,
            {
                "addRandomSuffix": False,
                "contentType": file.content_type or "application/octet-stream",
                "access": "public",
            }
        )
        
        return result["url"]
        
    def delete_file(file_path):
        """Delete a file from Vercel Blob storage"""
        if not file_path:
            return
            
        try:
            # Extract the blob ID from the URL
            if "/blob/" in file_path:
                blob_id = file_path.split("/blob/")[1].split("?")[0]
                blob_del(blob_id)
        except Exception as e:
            print(f"Error deleting file from blob storage: {e}")
            
    def get_file_url(filename):
        """Get the URL for a file in Vercel Blob storage"""
        return filename
        
else:
    # Development: Use local storage
    print("Using local file storage")
    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    def upload_file(file, upload_dir="uploads"):
        """Save a file to local storage"""
        if not file or not file.filename:
            return None
            
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(save_path)
        
        return f"uploads/{unique_filename}"
        
    def delete_file(file_path):
        """Delete a file from local storage"""
        if not file_path:
            return
            
        try:
            # Extract filename from path
            if file_path.startswith("uploads/"):
                filename = file_path[8:]  # Remove "uploads/" prefix
                full_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(full_path):
                    os.remove(full_path)
        except Exception as e:
            print(f"Error deleting file from local storage: {e}")
            
    def get_file_url(filename):
        """Get the URL for a file in local storage"""
        return f"/static/uploads/{filename}"

# Helper function to ensure upload directory exists
def ensure_upload_dir():
    """Ensure the upload directory exists"""
    if BLOB_READ_WRITE_TOKEN is None:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Export the main functions
__all__ = ['upload_file', 'delete_file', 'get_file_url', 'ensure_upload_dir']