from datetime import date, time
from decimal import Decimal

from app.bot.formatters import (
    format_confirmation,
    format_details,
    format_money,
    format_ranking,
    format_results,
)
from app.domain.models import (
    EventConstraint,
    HotelMode,
    HotelPreferences,
    HotelRateDetails,
    HotelRoomDetails,
    OfferDetails,
    SortPreference,
    TimeWindow,
    TransportMode,
    TransportPreferences,
    TripComponent,
    TripRequest,
    TripSearchFailure,
    TripSearchResult,
)
from tests.unit.test_trip_handoff import search_result


def test_confirmation_escapes_user_controlled_text() -> None:
    request = TripRequest(
        origin="<b>Москва</b>",
        destination="Казань & район",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel=HotelPreferences(mode=HotelMode.REQUIRED),
        transport=TransportPreferences(allowed_modes={TransportMode.RAIL}),
    )

    rendered = format_confirmation(request)

    assert "<b>Москва</b>" not in rendered
    assert "&lt;b&gt;Москва&lt;/b&gt;" in rendered
    assert "Казань &amp; район" in rendered
    assert "Транспорт: поезд" in rendered
    assert "rail" not in rendered


def test_all_ranking_winners_are_explained_as_one_option() -> None:
    title, explanation = format_ranking(
        frozenset(
            {
                SortPreference.CHEAPEST,
                SortPreference.FASTEST,
                SortPreference.BALANCED,
            }
        )
    )

    assert title == "Оптимальный и самый бюджетный"
    assert explanation == (
        "Это самый бюджетный и быстрый вариант, при этом лучший по балансу условий."
    )
    assert "cheapest" not in title + explanation


def test_two_winning_criteria_are_joined_in_plain_russian() -> None:
    title, explanation = format_ranking(
        frozenset({SortPreference.CHEAPEST, SortPreference.FASTEST})
    )

    assert title == "Самый бюджетный и быстрый"
    assert explanation == (
        "Этот вариант одновременно самый бюджетный и требует меньше всего времени в дороге "
        "среди найденных."
    )


def test_money_is_formatted_for_humans() -> None:
    assert format_money(Decimal("1322.0"), "RUB") == "1 322 ₽"
    assert format_money(Decimal("18875.22"), "RUB") == "18 875,22 ₽"


def test_result_names_transport_for_each_direction() -> None:
    rendered = format_results(search_result())

    assert "Туда: поезд ·" in rendered
    assert "Обратно: поезд ·" in rendered
    assert "В дороге:" in rendered
    assert "В городе:" in rendered
    assert "Проживание: 22 августа — 23 августа, 1 ночь" in rendered
    assert "rail" not in rendered
    assert "В известную сумму входят: дорога туда, дорога обратно, отель" in rendered
    assert "Не включено: питание, городской транспорт и активности" in rendered


def test_no_hotel_result_explicitly_states_that_accommodation_is_absent() -> None:
    base = search_result()
    option = base.options[0]
    combination = option.combination.model_copy(update={"hotel": None, "stay": None})
    result = base.model_copy(
        update={"options": (option.model_copy(update={"combination": combination}),)}
    )

    rendered = format_results(result)

    assert "Отель: не нужен" in rendered


def test_confirmation_shows_every_constraint_that_affects_search() -> None:
    request = TripRequest(
        origin="Москва",
        destination="Казань",
        departure_date=date(2026, 8, 21),
        return_date=date(2026, 8, 23),
        transport=TransportPreferences(
            allowed_modes={TransportMode.RAIL},
            max_transfers=1,
            departure_window=TimeWindow(from_time=time(8), to_time=time(12)),
            return_window=TimeWindow(from_time=time(16)),
        ),
        hotel=HotelPreferences(
            mode=HotelMode.REQUIRED,
            stars_min=4,
            rating_min=Decimal("8.5"),
            meal_preferences={"breakfast"},
            free_cancellation_required=True,
            required_amenities={"wifi"},
        ),
        event=EventConstraint(
            date=date(2026, 8, 22),
            start_time=time(20),
            end_time=time(1),
            end_date=date(2026, 8, 23),
        ),
        sort=SortPreference.FASTEST,
    )

    rendered = format_confirmation(request)

    assert "Пересадки: не более 1" in rendered
    assert "с 08:00 до 12:00" in rendered
    assert "после 16:00" in rendered
    assert "от 4★" in rendered
    assert "бесплатная отмена" in rendered
    assert "до 23 августа 01:00" in rendered
    assert "минимум времени в дороге" in rendered


def test_no_results_uses_typed_failure_message_and_escapes_provider_text() -> None:
    base = search_result()
    result = TripSearchResult(
        request=base.request,
        failures=(
            TripSearchFailure(
                category="no_transport",
                component="transport",
                retryable=False,
                user_message="Нет <подходящих> вариантов",
            ),
        ),
        searched_at=base.searched_at,
    )

    rendered = format_results(result)

    assert "поездка пока не складывается" in rendered
    assert "&lt;подходящих&gt;" in rendered


def test_details_render_verified_facts_and_localized_rate_price() -> None:
    details = OfferDetails(
        product_type="hotels",
        title="Тест & Hotel",
        check_in_time=time(14),
        check_out_time=time(12),
        amenities=("Wi-Fi", "Парковка"),
        facts=("Подтверждённый факт",),
        rooms=(
            HotelRoomDetails(
                name="Стандарт",
                rates=(HotelRateDetails(price=Decimal("1234.5"), currency="RUB"),),
            ),
        ),
    )

    rendered = format_details(TripComponent.HOTEL, details)

    assert "Тест &amp; Hotel" in rendered
    assert "Заезд после 14:00" in rendered
    assert "Подтверждённый факт" in rendered
    assert "1 234,50 ₽" in rendered
    assert "Пример доступного тарифа" in rendered


def test_generic_provider_transport_title_is_localized() -> None:
    details = OfferDetails(product_type="bus", title="bus offer")

    rendered = format_details(TripComponent.OUTBOUND, details)

    assert "Автобусный билет" in rendered
    assert "bus offer" not in rendered


def test_railway_provider_title_is_never_exposed_to_user() -> None:
    details = OfferDetails(product_type="railway", title="railway offer")

    rendered = format_details(TripComponent.OUTBOUND, details)

    assert "Билет на поезд" in rendered
    assert "railway offer" not in rendered
