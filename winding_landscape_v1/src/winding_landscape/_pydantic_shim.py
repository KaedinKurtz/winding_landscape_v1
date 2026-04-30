"""Tiny pydantic v2 compatibility shim.

Used only when real pydantic isn't importable -- this lets the math-heavy test
suite run in environments without pydantic available, while the production
install (which has real pydantic) gets full validation.

Implements only the subset used by ``config.py``:
  - BaseModel with model_config = ConfigDict(extra="forbid")
  - Field(default=..., default_factory=..., gt=..., ge=..., le=..., lt=...)
  - field_validator decorator
  - .model_dump(), .model_dump_json(), .model_validate(), .model_copy(update=)

Validation is intentionally limited (types, gt/ge/le/lt bounds, user-supplied
@field_validators). Errors raise ValueError.
"""

from __future__ import annotations

import dataclasses
import json
import types as _types
import typing as _typing
from typing import Any, Callable, get_args, get_origin, get_type_hints


_MISSING = object()  # sentinel for "no default supplied"


class ConfigDict(dict):
    """Mimics pydantic.ConfigDict -- a plain dict the user passes settings into."""


class _FieldInfo:
    __slots__ = ("default", "default_factory", "gt", "ge", "le", "lt")

    def __init__(
        self,
        *,
        default: Any = _MISSING,
        default_factory: Callable[[], Any] | None = None,
        gt: float | None = None,
        ge: float | None = None,
        le: float | None = None,
        lt: float | None = None,
    ) -> None:
        self.default = default
        self.default_factory = default_factory
        self.gt = gt
        self.ge = ge
        self.le = le
        self.lt = lt

    def has_default(self) -> bool:
        return self.default is not _MISSING or self.default_factory is not None

    def get_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


def Field(  # noqa: N802 -- match pydantic capitalization
    default: Any = _MISSING,
    *,
    default_factory: Callable[[], Any] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
    lt: float | None = None,
) -> Any:
    return _FieldInfo(
        default=default,
        default_factory=default_factory,
        gt=gt, ge=ge, le=le, lt=lt,
    )


def field_validator(field_name: str) -> Callable:
    """Decorator: register the wrapped function as a validator for ``field_name``."""
    def decorator(func: Callable) -> Callable:
        target = func.__func__ if isinstance(func, classmethod) else func
        existing = getattr(target, "_shim_validator_fields", [])
        existing.append(field_name)
        target._shim_validator_fields = existing
        return func
    return decorator


class BaseModel:
    """Minimal pydantic-v2-shaped BaseModel."""

    model_config: ConfigDict = ConfigDict()

    def __init__(self, **data: Any) -> None:
        hints = get_type_hints(type(self))
        if self.model_config.get("extra") == "forbid":
            unknown = set(data) - set(hints) - {"model_config"}
            if unknown:
                raise ValueError(
                    f"{type(self).__name__}: unexpected fields {sorted(unknown)}"
                )

        for field_name, field_type in hints.items():
            if field_name == "model_config":
                continue
            class_default = type(self).__dict__.get(field_name, _MISSING)
            if class_default is _MISSING:
                for base in type(self).__mro__[1:]:
                    if field_name in base.__dict__:
                        class_default = base.__dict__[field_name]
                        break
            field_info: _FieldInfo | None = (
                class_default if isinstance(class_default, _FieldInfo) else None
            )

            if field_name in data:
                raw_value = data[field_name]
            elif field_info is not None and field_info.has_default():
                raw_value = field_info.get_default()
            elif class_default is not _MISSING and not isinstance(class_default, _FieldInfo):
                raw_value = class_default
            else:
                raise ValueError(
                    f"{type(self).__name__}: missing required field '{field_name}'"
                )

            value = _coerce_value(raw_value, field_type)

            if field_info is not None:
                _check_bounds(field_name, value, field_info)

            # Run @field_validators declared on any class in MRO.
            for cls in type(self).__mro__:
                for attr_name, attr in cls.__dict__.items():
                    target = attr.__func__ if isinstance(attr, classmethod) else attr
                    fnames = getattr(target, "_shim_validator_fields", None)
                    if fnames and field_name in fnames:
                        value = _call_validator(attr, type(self), value)

            object.__setattr__(self, field_name, value)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "BaseModel":
        return cls(**data)

    def model_dump(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        hints = get_type_hints(type(self))
        for field_name in hints:
            if field_name == "model_config":
                continue
            out[field_name] = _to_jsonable(getattr(self, field_name))
        return out

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), default=str)

    def model_copy(self, *, update: dict[str, Any] | None = None) -> "BaseModel":
        data = self.model_dump()
        if update:
            data.update(update)
        return type(self)(**data)


def _coerce_value(value: Any, type_hint: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        if isinstance(value, type_hint):
            return value
        if isinstance(value, dict):
            return type_hint(**value)
        raise ValueError(f"Expected {type_hint.__name__} or dict, got {type(value).__name__}")

    if origin is _typing.Literal:
        if value not in args:
            raise ValueError(f"Value {value!r} not in allowed literals {args}")
        return value

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Expected tuple/list, got {type(value).__name__}")
        return tuple(value)

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value).__name__}")
        return value

    if origin in (_typing.Union, _types.UnionType):
        return value

    if type_hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value


def _check_bounds(name: str, value: Any, fi: _FieldInfo) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if fi.gt is not None and not value > fi.gt:
        raise ValueError(f"{name}={value} must be > {fi.gt}")
    if fi.ge is not None and not value >= fi.ge:
        raise ValueError(f"{name}={value} must be >= {fi.ge}")
    if fi.le is not None and not value <= fi.le:
        raise ValueError(f"{name}={value} must be <= {fi.le}")
    if fi.lt is not None and not value < fi.lt:
        raise ValueError(f"{name}={value} must be < {fi.lt}")


def _call_validator(validator: Any, cls: type, value: Any) -> Any:
    if isinstance(validator, classmethod):
        return validator.__func__(cls, value)
    if callable(validator):
        try:
            return validator(cls, value)
        except TypeError:
            return validator(value)
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value
