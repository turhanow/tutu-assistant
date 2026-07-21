import hashlib
import json
from pathlib import Path

SNAPSHOT = Path("tests/fixtures/tutu_mcp/tools-list.json")
EXPECTED_HASH = "1a00e172686c927cf2d540c27049abe79764ec0635c232c5b236eaec6ee751bd"


def test_tutu_contract_snapshot_is_complete_and_unchanged() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    pages = snapshot["pages"]
    tools = [tool for page in pages for tool in page["tools"]]
    names = [tool["name"] for tool in tools]
    payload = json.dumps(
        pages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert len(tools) == 16
    assert len(names) == len(set(names))
    assert all(tool["inputSchema"] for tool in tools)
    assert digest == snapshot["sha256"] == EXPECTED_HASH


def test_minimal_smoke_fixtures_are_present_and_anonymized() -> None:
    fixture_dir = Path("tests/fixtures/tutu_mcp/fixtures")
    expected = {
        "transport-search.json",
        "hotel-search.json",
        "checkout-link.json",
        "hotel-details.json",
        "rail-search.json",
        "avia-search.json",
    }

    for name in expected:
        content = (fixture_dir / name).read_text(encoding="utf-8")
        payload = json.loads(content)
        assert isinstance(payload, dict)
        assert "fixture" in content
