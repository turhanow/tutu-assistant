from app.voice import (
    BOT_DESCRIPTION,
    BOT_SHORT_DESCRIPTION,
    BRAND_NAME,
    DELIGHT_COPY,
    DISCOVERY_START,
    FORBIDDEN_SLANG,
    HELP,
    KNOWN_START,
    PRIVACY,
    PROGRESS_COPY,
    ROUTER_START,
    Voice,
)


def test_brand_profile_fits_telegram_contract() -> None:
    assert BRAND_NAME == "Ту-да и обратно"
    assert 1 <= len(BRAND_NAME) <= 64
    assert 1 <= len(BOT_SHORT_DESCRIPTION) <= 120
    assert 1 <= len(BOT_DESCRIPTION) <= 512


def test_runtime_voice_copy_avoids_slang_and_exclamation_marks() -> None:
    copy = "\n".join(
        (
            ROUTER_START,
            KNOWN_START,
            DISCOVERY_START,
            HELP,
            PRIVACY,
            *PROGRESS_COPY.values(),
            *DELIGHT_COPY.values(),
        )
    ).casefold()

    assert "!" not in copy
    assert not any(word in copy for word in FORBIDDEN_SLANG)


def test_controlled_delight_appears_at_most_once_per_dialog() -> None:
    voice = Voice(controlled_delight_enabled=True)
    user_data = {}

    first = voice.progress("known_search", user_data)
    repeated = voice.progress("known_search", user_data)
    another = voice.progress("discovery_verify", user_data)

    assert DELIGHT_COPY["known_search"] in first
    assert DELIGHT_COPY["known_search"] not in repeated
    assert DELIGHT_COPY["discovery_verify"] not in another


def test_voice_feature_flags_provide_instant_safe_rollback() -> None:
    legacy = Voice(tone_v2_enabled=False, controlled_delight_enabled=True)
    user_data = {}

    assert legacy.version == "voice_v1"
    assert "сценарий" in legacy.router_start
    assert "экскурсию по пересадкам" not in legacy.progress("known_search", user_data)
    assert user_data == {}


def test_sensitive_copy_has_no_delight() -> None:
    voice = Voice(controlled_delight_enabled=True)
    sensitive = "\n".join((PRIVACY, HELP))

    assert not any(line in sensitive for line in DELIGHT_COPY.values())
    assert voice.analytics_dimensions == {"voice_version": "voice_v2"}


def test_moscow_only_discovery_scope_is_disclosed_before_intake() -> None:
    assert "для выезда из Москвы" in ROUTER_START
    assert "только для выезда из Москвы" in DISCOVERY_START
    assert "MVP" not in DISCOVERY_START
    assert "/newtrip" in DISCOVERY_START
