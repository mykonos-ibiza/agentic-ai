import socket
import ipaddress
from urllib.parse import urlparse, urljoin
from typing import Tuple, Optional

__all__ = [
    "is_ip_disallowed",
    "validate_public_http_url",
    "safe_follow_redirects_requests",
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