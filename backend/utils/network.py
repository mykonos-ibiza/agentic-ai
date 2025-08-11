import socket
import ipaddress
from urllib.parse import urlparse, urljoin
from typing import Tuple, Optional

__all__ = [
    "is_ip_disallowed",
    "validate_public_http_url",
    "safe_follow_redirects_requests",
    "download_image_requests",
    "async_download_image_httpx",
]


def is_ip_disallowed(ip: str) -> bool:
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


def validate_public_http_url(url: str, allowed_ports: Tuple[int, ...] = (80, 443)) -> str:
    """Validate that the URL is http(s), uses allowed ports, and resolves only to public IPs.
    Returns a normalized URL string. Raises ValueError on failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("User info in URL is not allowed")

    port: Optional[int] = parsed.port
    if port is not None and port not in allowed_ports:
        raise ValueError("Port not allowed")

    try:
        resolved = socket.getaddrinfo(parsed.hostname, port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError("Hostname could not be resolved")

    ips = {item[4][0] for item in resolved if item and item[4]}
    if not ips:
        raise ValueError("No IPs resolved for hostname")
    for ip in ips:
        if is_ip_disallowed(ip):
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


def safe_follow_redirects_requests(session, url: str, headers: Optional[dict] = None, max_redirects: int = 3) -> str:
    """Perform HEAD requests to follow redirects manually with validation at each hop.
    Returns the final validated URL or raises on failure.
    """
    headers = headers or {}
    current_url = validate_public_http_url(url)
    for _ in range(max_redirects):
        resp = session.head(current_url, timeout=10, headers=headers, allow_redirects=False)
        # Not a redirect
        if resp.status_code < 300 or resp.status_code >= 400 or 'Location' not in resp.headers:
            return current_url
        next_url = urljoin(current_url, resp.headers['Location'])
        current_url = validate_public_http_url(next_url)
    raise ValueError("Too many redirects")


def download_image_requests(url: str, max_size_bytes: int, headers: Optional[dict] = None, max_redirects: int = 3) -> Tuple[bytes, str]:
    """Download an image using requests with SSRF protections and size/content-type checks.
    Returns (image_bytes, mime_type).
    """
    import requests

    headers = headers or {"User-Agent": "Mozilla/5.0"}
    with requests.Session() as session:
        final_url = safe_follow_redirects_requests(session, url, headers=headers, max_redirects=max_redirects)
        resp = session.get(final_url, timeout=10, headers=headers, allow_redirects=False, stream=True)
        resp.raise_for_status()

        mime_type = resp.headers.get("Content-Type", "")
        if not mime_type.startswith("image/"):
            raise ValueError(f"URL does not point to an image (Content-Type: {mime_type}): {final_url}")

        cl_header = resp.headers.get("Content-Length")
        if cl_header is not None:
            try:
                if int(cl_header) > max_size_bytes:
                    raise ValueError("Image is too large")
            except (TypeError, ValueError):
                pass

        downloaded = 0
        chunks = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_size_bytes:
                raise ValueError("Downloaded image exceeds maximum allowed size")
            chunks.extend(chunk)
        return bytes(chunks), mime_type


async def async_download_image_httpx(url: str, max_size_bytes: int, headers: Optional[dict] = None, max_redirects: int = 0) -> Tuple[bytes, str]:
    """Download an image using httpx with SSRF protections and size/content-type checks.
    Does not follow redirects automatically; caller can set max_redirects>0 to implement manual logic later.
    Returns (image_bytes, mime_type).
    """
    import httpx

    headers = headers or {"User-Agent": "Mozilla/5.0"}
    safe_url = validate_public_http_url(url)
    async with httpx.AsyncClient(follow_redirects=False, timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(safe_url, headers=headers)
        resp.raise_for_status()
        mime_type = resp.headers.get("Content-Type", "")
        if not mime_type.startswith("image/"):
            raise ValueError(f"URL does not point to an image (Content-Type: {mime_type}): {safe_url}")
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            try:
                if int(cl) > max_size_bytes:
                    raise ValueError("Image is too large")
            except ValueError:
                pass
        content = bytearray()
        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > max_size_bytes:
                raise ValueError("Downloaded image exceeds maximum allowed size")
        return bytes(content), mime_type