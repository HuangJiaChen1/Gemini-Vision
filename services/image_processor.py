import base64
import io
from PIL import Image
from typing import Tuple, Optional


class ImageProcessor:
    """Handles image validation, processing, and conversion"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSION = 1024  # Max width or height for compression
    TARGET_SIZE = 1 * 1024 * 1024  # 1MB target for Gemini API
    ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}

    @staticmethod
    def validate_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validate image format and size
        Returns: (is_valid, error_message)
        """
        if len(image_bytes) > ImageProcessor.MAX_FILE_SIZE:
            return False, "Photo too big! Try a smaller one."

        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.format not in ImageProcessor.ALLOWED_FORMATS:
                return False, "This isn't a photo! Please choose a .jpg or .png file."
            return True, None
        except Exception:
            return False, "Can't open this file! Make sure it's a photo."

    @staticmethod
    def process_base64(base64_data: str) -> bytes:
        """
        Convert base64 string to bytes
        Handles both with and without data URI prefix
        """
        # Remove data URI prefix if present (data:image/jpeg;base64,...)
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]

        return base64.b64decode(base64_data)

    @staticmethod
    def resize_image(image_bytes: bytes) -> bytes:
        """
        Resize image if larger than MAX_DIMENSION or TARGET_SIZE
        Returns compressed image bytes
        """
        # If already under target size, return as-is
        if len(image_bytes) <= ImageProcessor.TARGET_SIZE:
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes))
            original_format = img.format or 'JPEG'

            # Resize if dimensions are too large
            width, height = img.size
            if width > ImageProcessor.MAX_DIMENSION or height > ImageProcessor.MAX_DIMENSION:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = ImageProcessor.MAX_DIMENSION
                    new_height = int((ImageProcessor.MAX_DIMENSION / width) * height)
                else:
                    new_height = ImageProcessor.MAX_DIMENSION
                    new_width = int((ImageProcessor.MAX_DIMENSION / height) * width)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save with compression
            output = io.BytesIO()
            save_format = 'JPEG' if original_format in ['JPEG', 'JPG'] else original_format

            # Convert RGBA to RGB for JPEG
            if save_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img

            img.save(output, format=save_format, quality=85, optimize=True)
            return output.getvalue()

        except Exception as e:
            # If resize fails, return original
            print(f"Warning: Could not resize image: {e}")
            return image_bytes

    @staticmethod
    def get_mime_type(image_bytes: bytes) -> str:
        """Detect MIME type from image bytes"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            format_lower = img.format.lower() if img.format else 'jpeg'
            return f"image/{format_lower}"
        except Exception:
            return "image/jpeg"  # Default fallback
