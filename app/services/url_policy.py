"""Central hostname policy for externally supplied URLs."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def require_allowed_https_url(
    value: str,
    *,
    allowed_hosts: frozenset[str],
    allow_subdomains: bool = False,
) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must be credential-free HTTPS")
    host = parsed.hostname.casefold().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP literal URLs are not allowed")
    allowed = host in allowed_hosts or (
        allow_subdomains and any(host.endswith(f".{candidate}") for candidate in allowed_hosts)
    )
    if not allowed:
        raise ValueError("URL host is not allowlisted")
