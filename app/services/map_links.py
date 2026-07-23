"""Deterministic Yandex Maps search links for places without fabricated organization IDs."""

from urllib.parse import quote, urlencode


def build_yandex_maps_search_url(
    *,
    name: str,
    address: str | None = None,
    city: str | None = None,
    region: str | None = None,
) -> str:
    """Build a canonical text-search URL and avoid repeating locality components."""
    parts = [name.strip()]
    location_text = (address or "").strip()
    if location_text:
        parts.append(location_text)
    location_folded = location_text.casefold()
    if city and city.strip() and city.casefold() not in location_folded:
        parts.append(city.strip())
    if region and region.strip() and region.casefold() not in location_folded:
        parts.append(region.strip())
    query = ", ".join(part for part in parts if part)
    return "https://yandex.ru/maps/?" + urlencode({"text": query}, quote_via=quote)
