"""Typed errors crossing domain and application boundaries."""


class TutuAssistantError(Exception):
    """Base class for expected application failures."""


class TripValidationError(TutuAssistantError):
    """The requested trip is structurally invalid."""


class UnsupportedTripError(TutuAssistantError):
    """The request is valid but outside the MVP scope."""


class ProviderError(TutuAssistantError):
    """Base class for failures reported by an external provider."""


class ProviderTransientError(ProviderError):
    """A retryable provider connection, timeout, or rate-limit failure."""


class ProviderContractError(ProviderError):
    """The live provider contract differs from the captured contract."""


class ProviderResponseError(ProviderError):
    """A provider response cannot be mapped safely into the domain."""


class ProviderToolError(ProviderError):
    """A provider tool returned an application-level error."""


class CheckoutError(ProviderError):
    """A safe handoff URL could not be produced."""


class LlmParseError(TutuAssistantError):
    """The configured LLM did not produce a valid trip extraction."""


class LlmProviderError(TutuAssistantError):
    """The configured LLM provider is temporarily unavailable."""


class CatalogError(TutuAssistantError):
    """Base class for destination catalog failures."""


class CatalogContractError(CatalogError):
    """The versioned catalog cannot be validated safely."""


class CatalogItemNotFoundError(CatalogError):
    """The requested destination does not exist in the active catalog."""


class UnsupportedOriginError(CatalogError):
    """Destination discovery is not available from the requested origin."""
