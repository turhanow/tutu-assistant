def test_composition_root_is_importable_without_runtime_secrets() -> None:
    from app import main

    assert callable(main.main)


def test_domain_does_not_import_adapter_packages() -> None:
    import app.domain.models as models

    imported_names = set(models.__dict__)
    assert not {"telegram", "mcp", "httpx"} & imported_names


def test_discovery_domain_does_not_import_external_adapters() -> None:
    import app.domain.content_models as content_models
    import app.domain.discovery_models as discovery_models
    import app.domain.product_models as product_models

    for module in (content_models, discovery_models, product_models):
        imported_names = set(module.__dict__)
        assert not {"telegram", "openai", "mcp", "httpx"} & imported_names
