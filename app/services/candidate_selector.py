"""Deterministic pre-network destination candidate selection."""

from __future__ import annotations

from decimal import Decimal

from app.domain.content_models import DestinationProfile
from app.domain.discovery_models import (
    CandidateShortlist,
    DestinationCandidate,
    DiscoveryRequest,
)
from app.domain.errors import UnsupportedOriginError
from app.ports.catalog import DestinationCatalog

SCORE_VERSION = "candidate_v1"
MAX_POOL_SIZE = 12
MAX_SHORTLIST_SIZE = 8

TAG_ALIASES = {
    "architecture": "architecture",
    "архитектура": "architecture",
    "старый город": "architecture",
    "culture": "culture",
    "культура": "culture",
    "history": "history",
    "история": "history",
    "museum": "museum",
    "museums": "museum",
    "музей": "museum",
    "музеи": "museum",
    "nature": "nature",
    "природа": "nature",
    "gastronomy": "gastronomy",
    "гастрономия": "gastronomy",
    "еда": "gastronomy",
    "walking": "walking",
    "прогулки": "walking",
    "смена обстановки": "walking",
    "science": "science",
    "наука": "science",
    "space": "space",
    "космос": "space",
    "craft": "craft",
    "ремесла": "craft",
    "river": "river",
    "река": "river",
    "реки": "river",
    "рекой": "river",
    "у реки": "river",
    "у воды": "river",
    "набережная": "river",
    "набережные": "river",
    "relaxed": "relaxed",
    "спокойный отдых": "relaxed",
    "спокойно": "relaxed",
    "без суеты": "relaxed",
}

TAG_LABELS = {
    "architecture": "архитектура",
    "culture": "культура",
    "history": "история",
    "museum": "музеи",
    "nature": "природа",
    "gastronomy": "гастрономия",
    "walking": "прогулки",
    "science": "наука",
    "space": "космос",
    "craft": "ремёсла",
    "river": "набережные и река",
    "relaxed": "спокойный темп",
}


class CandidateSelector:
    def __init__(self, catalog: DestinationCatalog) -> None:
        self._catalog = catalog

    async def select(
        self,
        request: DiscoveryRequest,
        *,
        limit: int = MAX_SHORTLIST_SIZE,
    ) -> CandidateShortlist:
        supported_origins = {item.casefold() for item in self._catalog.pilot_origins}
        if request.origin.casefold() not in supported_origins:
            supported = ", ".join(sorted(self._catalog.pilot_origins))
            raise UnsupportedOriginError(
                f"Подбор направлений пока работает только из: {supported}. "
                "Для конкретного маршрута из другого города используйте /newtrip."
            )
        bounded_limit = min(max(limit, 0), MAX_SHORTLIST_SIZE)
        if bounded_limit == 0:
            return CandidateShortlist(
                request=request,
                candidates=(),
                catalog_version=self._catalog.version,
                score_version=SCORE_VERSION,
            )
        profiles = await self._catalog.find_candidates(request, limit=MAX_POOL_SIZE)
        scored = [
            candidate
            for profile in profiles
            if (candidate := self._score(request, profile)) is not None
        ]
        scored.sort(
            key=lambda item: (
                -item.match_score,
                item.destination.destination_id,
            )
        )
        selected = _diversify(scored, limit=bounded_limit)
        return CandidateShortlist(
            request=request,
            candidates=tuple(selected),
            catalog_version=self._catalog.version,
            score_version=SCORE_VERSION,
        )

    @staticmethod
    def _score(
        request: DiscoveryRequest,
        profile: DestinationProfile,
    ) -> DestinationCandidate | None:
        days = (request.dates.end - request.dates.start).days + 1
        experience_values = request.experience.motives | request.experience.interests
        if request.experience.pace is not None:
            experience_values = experience_values | {request.experience.pace.value}
        requested_tags = _canonical_tags(experience_values)
        profile_tags = _canonical_tags(profile.experience_tags)
        overlap = requested_tags & profile_tags
        tag_score = (
            Decimal("0.5")
            if not requested_tags
            else Decimal(len(overlap)) / Decimal(len(requested_tags))
        )
        requested_hours = Decimal(days * 24)
        duration = profile.typical_visit_duration
        total_microseconds = (
            duration.days * 86_400 + duration.seconds
        ) * 1_000_000 + duration.microseconds
        profile_hours = Decimal(total_microseconds) / Decimal(3_600_000_000)
        duration_delta = abs(profile_hours - requested_hours)
        duration_score = max(Decimal(0), Decimal(1) - duration_delta / Decimal(72))
        evidence_score = Decimal(1) if profile.evidence_ids else Decimal(0)
        score = (
            Decimal("0.65") * tag_score
            + Decimal("0.20") * duration_score
            + Decimal("0.15") * evidence_score
        ).quantize(Decimal("0.0001"))

        reasons: list[str] = []
        if profile.activity_highlights:
            reasons.append(
                "Под запрос подходят: " + "; ".join(profile.activity_highlights[:2])
            )
        if overlap:
            labels = ", ".join(TAG_LABELS.get(tag, tag) for tag in sorted(overlap)[:2])
            reasons.append(f"Подходит по интересам: {labels}")
        if not reasons:
            reasons.append(f"Можно исследовать {profile.name} в своём темпе")
        return DestinationCandidate(
            destination=profile,
            match_score=score,
            match_reasons=tuple(reasons[:3]),
        )


def _canonical_tags(values: frozenset[str]) -> frozenset[str]:
    return frozenset(
        TAG_ALIASES.get(value.strip().casefold(), value.strip().casefold())
        for value in values
        if value.strip()
    )


def _diversify(
    candidates: list[DestinationCandidate],
    *,
    limit: int,
) -> list[DestinationCandidate]:
    selected: list[DestinationCandidate] = []
    deferred: list[DestinationCandidate] = []
    regions: set[str] = set()
    for candidate in candidates:
        region = candidate.destination.region.casefold()
        if region in regions:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        regions.add(region)
        if len(selected) == limit:
            return selected
    for candidate in deferred:
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected
