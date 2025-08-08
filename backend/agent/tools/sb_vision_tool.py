import os
import base64
import mimetypes
from typing import Optional, Tuple
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from agentpress.tool import ToolResult, openapi_schema, usage_example
from sandbox.tool_base import SandboxToolsBase
from agentpress.thread_manager import ThreadManager
import json
import requests
import socket
import ipaddress
from urllib.parse import urljoin

# Add common image MIME types if mimetypes module is limited
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/gif", ".gif")

# Maximum file size in bytes (e.g., 10MB for original, 5MB for compressed)
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_COMPRESSED_SIZE = 5 * 1024 * 1024

# Compression settings
DEFAULT_MAX_WIDTH = 1920
DEFAULT_MAX_HEIGHT = 1080
DEFAULT_JPEG_QUALITY = 85
DEFAULT_PNG_COMPRESS_LEVEL = 6

class SandboxVisionTool(SandboxToolsBase):
    """Tool for allowing the agent to 'see' images within the sandbox."""

    def __init__(self, project_id: str, thread_id: str, thread_manager: ThreadManager):
        super().__init__(project_id, thread_manager)
        self.thread_id = thread_id
        # Make thread_manager accessible within the tool instance
        self.thread_manager = thread_manager

    def compress_image(self, image_bytes: bytes, mime_type: str, file_path: str) -> Tuple[bytes, str]:
        """Compress an image to reduce its size while maintaining reasonable quality.
        
        Args:
            image_bytes: Original image bytes
            mime_type: MIME type of the image
            file_path: Path to the image file (for logging)
            
        Returns:
            Tuple of (compressed_bytes, new_mime_type)
        """
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_bytes))
            
            # Convert RGBA to RGB if necessary (for JPEG)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            if width > DEFAULT_MAX_WIDTH or height > DEFAULT_MAX_HEIGHT:
                ratio = min(DEFAULT_MAX_WIDTH / width, DEFAULT_MAX_HEIGHT / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"[SeeImage] Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Save to bytes with compression
            output = BytesIO()
            
            # Determine output format based on original mime type
            if mime_type == 'image/gif':
                # Keep GIFs as GIFs to preserve animation
                img.save(output, format='GIF', optimize=True)
                output_mime = 'image/gif'
            elif mime_type == 'image/png':
                # Compress PNG
                img.save(output, format='PNG', optimize=True, compress_level=DEFAULT_PNG_COMPRESS_LEVEL)
                output_mime = 'image/png'
            else:
                # Convert everything else to JPEG for better compression
                img.save(output, format='JPEG', quality=DEFAULT_JPEG_QUALITY, optimize=True)
                output_mime = 'image/jpeg'
            
            compressed_bytes = output.getvalue()
            
            # Log compression results
            original_size = len(image_bytes)
            compressed_size = len(compressed_bytes)
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"[SeeImage] Compressed '{file_path}' from {original_size / 1024:.1f}KB to {compressed_size / 1024:.1f}KB ({compression_ratio:.1f}% reduction)")
            
            return compressed_bytes, output_mime
            
        except Exception as e:
            print(f"[SeeImage] Failed to compress image: {str(e)}. Using original.")
            return image_bytes, mime_type

    def is_url(self, file_path: str) -> bool:
        """check if the file path is url"""
        parsed_url = urlparse(file_path)
        return parsed_url.scheme in ('http', 'https')
    
    def _is_ip_disallowed(self, ip: str) -> bool:
        """Return True if IP is private/loopback/link-local/multicast/reserved/unspecified."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            )
        except ValueError:
            return True

    def _validate_outbound_url(self, url: str) -> str:
        """Validate URL scheme, port, and DNS resolution to public IPs only. Return normalized URL."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http/https URLs are allowed")
        if not parsed.hostname:
            raise ValueError("URL must include a hostname")
        # Restrict ports to 80/443 (or default if None)
        port = parsed.port
        if port is not None and port not in (80, 443):
            raise ValueError("Only ports 80 and 443 are allowed")
        # Resolve hostname and ensure all resolved IPs are public
        try:
            # Use default port to avoid separate SRV lookups
            resolved = socket.getaddrinfo(parsed.hostname, port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror:
            raise ValueError("Hostname could not be resolved")
        ips = {item[4][0] for item in resolved if item and item[4]}
        if not ips:
            raise ValueError("No IPs resolved for hostname")
        for ip in ips:
            if self._is_ip_disallowed(ip):
                raise ValueError("URL resolves to a disallowed IP address")
        # Normalize by removing default ports
        netloc = parsed.hostname
        if port and port not in (80, 443):
            netloc = f"{parsed.hostname}:{port}"
        elif parsed.scheme == "http" and port == 80:
            netloc = parsed.hostname
        elif parsed.scheme == "https" and port == 443:
            netloc = parsed.hostname
        normalized = parsed._replace(netloc=netloc).geturl()
        return normalized

    def _safe_follow_redirects(self, session: requests.Session, url: str, headers: dict, max_redirects: int = 3) -> str:
        """Perform a HEAD request without following redirects; manually follow with validation."""
        current_url = self._validate_outbound_url(url)
        for _ in range(max_redirects):
            resp = session.head(current_url, timeout=10, headers=headers, allow_redirects=False)
            # If no redirect, return current URL
            if resp.status_code < 300 or resp.status_code >= 400 or 'Location' not in resp.headers:
                return current_url
            next_url = urljoin(current_url, resp.headers['Location'])
            current_url = self._validate_outbound_url(next_url)
        raise ValueError("Too many redirects")

    def download_image_from_url(self, url: str) -> Tuple[bytes, str]:
        """Download image from a URL with SSRF protections."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            with requests.Session() as session:
                # Resolve safe final URL via validated redirect handling
                final_url = self._safe_follow_redirects(session, url, headers)

                # Perform GET without auto-redirects
                get_resp = session.get(final_url, timeout=10, headers=headers, allow_redirects=False, stream=True)
                get_resp.raise_for_status()

                # Enforce content type is image
                mime_type = get_resp.headers.get('Content-Type', '')
                if not mime_type.startswith('image/'):
                    raise Exception(f"URL does not point to an image (Content-Type: {mime_type}): {final_url}")

                # Check Content-Length header if present
                cl_header = get_resp.headers.get('Content-Length')
                if cl_header is not None:
                    try:
                        content_length = int(cl_header)
                        if content_length > MAX_IMAGE_SIZE:
                            raise Exception(f"Image is too large ({(content_length)/(1024*1024):.2f}MB) for the maximum allowed size of {MAX_IMAGE_SIZE/(1024*1024):.2f}MB")
                    except (TypeError, ValueError):
                        # Ignore unparsable Content-Length; fall back to streaming cap
                        pass

                # Stream download with hard cap
                bytes_io = BytesIO()
                downloaded = 0
                for chunk in get_resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_IMAGE_SIZE:
                        raise Exception(f"Downloaded image is too large ({(downloaded)/(1024*1024):.2f}MB). Maximum allowed size of {MAX_IMAGE_SIZE/(1024*1024):.2f}MB")
                    bytes_io.write(chunk)
                image_bytes = bytes_io.getvalue()

                return image_bytes, mime_type
        except Exception as e:
            return self.fail_response(f"Failed to download image from URL: {str(e)}")
    
    @openapi_schema({
        "type": "function",
        "function": {
            "name": "see_image",
            "description": "Allows the agent to 'see' an image file located in the /workspace directory or from a URL. Provide either a relative path to a local image or the URL to an image. The image will be compressed before sending to reduce token usage. The image content will be made available in the next turn's context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Either a relative path to the image file within the /workspace directory (e.g., 'screenshots/image.png') or a URL to an image (e.g., 'https://example.com/image.jpg'). Supported formats: JPG, PNG, GIF, WEBP. Max size: 10MB."
                    }
                },
                "required": ["file_path"]
            }
        }
    })
    @usage_example('''
        <!-- Example: Request to see a local image named 'diagram.png' inside the 'docs' folder -->
        <function_calls>
        <invoke name="see_image">
        <parameter name="file_path">docs/diagram.png</parameter>
        </invoke>
        </function_calls>

        <!-- Example: Request to see an image from a URL -->
        <function_calls>
        <invoke name="see_image">
        <parameter name="file_path">https://example.com/image.jpg</parameter>
        </invoke>
        </function_calls>
        ''')
    async def see_image(self, file_path: str) -> ToolResult:
        """Reads an image file from local file system or from a URL, compresses it, converts it to base64, and adds it as a temporary message."""
        try:
            is_url = self.is_url(file_path)
            if is_url:
                try:
                    image_bytes, mime_type = self.download_image_from_url(file_path)
                    original_size = len(image_bytes)
                    cleaned_path = file_path
                except Exception as e:
                    return self.fail_response(f"Failed to download image from URL: {str(e)}")
            else:
                # Ensure sandbox is initialized
                await self._ensure_sandbox()

                # Clean and construct full path
                cleaned_path = self.clean_path(file_path)
                full_path = f"{self.workspace_path}/{cleaned_path}"

                # Check if file exists and get info
                try:
                    file_info = await self.sandbox.fs.get_file_info(full_path)
                    if file_info.is_dir:
                        return self.fail_response(f"Path '{cleaned_path}' is a directory, not an image file.")
                except Exception as e:
                    return self.fail_response(f"Image file not found at path: '{cleaned_path}'")

                # Check file size
                if file_info.size > MAX_IMAGE_SIZE:
                    return self.fail_response(f"Image file '{cleaned_path}' is too large ({file_info.size / (1024*1024):.2f}MB). Maximum size is {MAX_IMAGE_SIZE / (1024*1024)}MB.")

                # Read image file content
                try:
                    image_bytes = await self.sandbox.fs.download_file(full_path)
                except Exception as e:
                    return self.fail_response(f"Could not read image file: {cleaned_path}")

                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(full_path)
                if not mime_type or not mime_type.startswith('image/'):
                    # Basic fallback based on extension if mimetypes fails
                    ext = os.path.splitext(cleaned_path)[1].lower()
                    if ext == '.jpg' or ext == '.jpeg': mime_type = 'image/jpeg'
                    elif ext == '.png': mime_type = 'image/png'
                    elif ext == '.gif': mime_type = 'image/gif'
                    elif ext == '.webp': mime_type = 'image/webp'
                    else:
                        return self.fail_response(f"Unsupported or unknown image format for file: '{cleaned_path}'. Supported: JPG, PNG, GIF, WEBP.")
                
                original_size = file_info.size
            

            # Compress the image
            compressed_bytes, compressed_mime_type = self.compress_image(image_bytes, mime_type, cleaned_path)
            
            # Check if compressed image is still too large
            if len(compressed_bytes) > MAX_COMPRESSED_SIZE:
                return self.fail_response(f"Image file '{cleaned_path}' is still too large after compression ({len(compressed_bytes) / (1024*1024):.2f}MB). Maximum compressed size is {MAX_COMPRESSED_SIZE / (1024*1024)}MB.")

            # Convert to base64
            base64_image = base64.b64encode(compressed_bytes).decode('utf-8')

            # Prepare the temporary message content
            image_context_data = {
                "mime_type": compressed_mime_type,
                "base64": base64_image,
                "file_path": cleaned_path, # Include path for context
                "original_size": original_size,
                "compressed_size": len(compressed_bytes)
            }

            # Add the temporary message using the thread_manager callback
            # Use a distinct type like 'image_context'
            await self.thread_manager.add_message(
                thread_id=self.thread_id,
                type="image_context", # Use a specific type for this
                content=image_context_data, # Store the dict directly
                is_llm_message=False # This is context generated by a tool
            )

            # Inform the agent the image will be available next turn
            return self.success_response(f"Successfully loaded and compressed the image '{cleaned_path}' (reduced from {original_size / 1024:.1f}KB to {len(compressed_bytes) / 1024:.1f}KB).")

        except Exception as e:
            return self.fail_response(f"An error occurred: {str(e)}") 