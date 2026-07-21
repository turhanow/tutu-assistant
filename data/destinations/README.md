# Destination catalog

`v1/catalog.json` — immutable pilot snapshot for trips from Moscow. Runtime loads and validates
the complete file at startup; invalid or missing data fails closed before discovery is enabled.

Catalog content is deliberately stable: it records destination themes and well-known places,
not opening hours, live prices or event availability. Activity duration is a planning estimate,
not a provider promise. Transport, hotel, price and availability are verified separately through
Tutu MCP.

Every place has at least one HTTPS evidence reference. Images are absent until licensing,
attribution and delivery rules are implemented. Update requires a new catalog version and passing:

```bash
python scripts/validate_destination_catalog.py
pytest -q tests/unit/test_file_catalog.py tests/unit/test_catalog_contract.py
```
