"""Error contract (MIB section 11 / ICIB section 9).

Every error carries: code, category, severity, capability, message and a
recommended action. Nothing is ever swallowed silently (DIC-07).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ErrorCategory(str, enum.Enum):
    INPUT = "input"
    RUNTIME = "runtime"
    RESOURCE = "resource"
    VALIDATION = "validation"
    PACKAGING = "packaging"
    CONTRACT = "contract"
    TRUST = "trust"
    CONFIGURATION = "configuration"


class Severity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    code: str
    category: ErrorCategory
    severity: Severity
    message: str
    capability: str = "runtime"
    stage: str = "unknown"
    recommended_action: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "capability": self.capability,
            "stage": self.stage,
            "recommended_action": self.recommended_action,
            "details": dict(self.details),
        }


class LeanLMError(Exception):
    """Base class for every LeanLM failure. Always carries an ErrorRecord."""

    def __init__(self, record: ErrorRecord) -> None:
        super().__init__(f"[{record.code}] {record.message}")
        self.record = record

    @classmethod
    def of(
        cls,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        severity: Severity = Severity.ERROR,
        capability: str = "runtime",
        stage: str = "unknown",
        recommended_action: str = "",
        **details: Any,
    ) -> "LeanLMError":
        return cls(
            ErrorRecord(
                code=code,
                category=category,
                severity=severity,
                message=message,
                capability=capability,
                stage=stage,
                recommended_action=recommended_action,
                details=details,
            )
        )


class InputError(LeanLMError):
    pass


class ResourceError(LeanLMError):
    pass


class ContractError(LeanLMError):
    pass


class TrustError(LeanLMError):
    pass


class ConfigurationError(LeanLMError):
    pass


class InferenceError(LeanLMError):
    pass


class PackagingError(LeanLMError):
    pass


def input_error(code: str, message: str, **kw: Any) -> InputError:
    err = LeanLMError.of(code, message, category=ErrorCategory.INPUT, **kw)
    return InputError(err.record)


def resource_error(code: str, message: str, **kw: Any) -> ResourceError:
    err = LeanLMError.of(code, message, category=ErrorCategory.RESOURCE, **kw)
    return ResourceError(err.record)


def contract_error(code: str, message: str, **kw: Any) -> ContractError:
    err = LeanLMError.of(code, message, category=ErrorCategory.CONTRACT, **kw)
    return ContractError(err.record)


def trust_error(code: str, message: str, **kw: Any) -> TrustError:
    err = LeanLMError.of(code, message, category=ErrorCategory.TRUST,
                         severity=Severity.CRITICAL, **kw)
    return TrustError(err.record)


def config_error(code: str, message: str, **kw: Any) -> ConfigurationError:
    err = LeanLMError.of(code, message, category=ErrorCategory.CONFIGURATION, **kw)
    return ConfigurationError(err.record)


def inference_error(code: str, message: str, **kw: Any) -> InferenceError:
    err = LeanLMError.of(code, message, category=ErrorCategory.RUNTIME, **kw)
    return InferenceError(err.record)


def packaging_error(code: str, message: str, **kw: Any) -> PackagingError:
    err = LeanLMError.of(code, message, category=ErrorCategory.PACKAGING, **kw)
    return PackagingError(err.record)
