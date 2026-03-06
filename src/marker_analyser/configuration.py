"""Configuration models."""

from enum import Enum

from typing import Self

from pydantic import BaseModel, model_validator, ValidationError

FITTING_PARAMS = ["Lp", "Lc", "St", "f_offset", "kT"]


class FitType(Enum):
    """Enum to specify whether a fit is individual or global."""

    INDIVIDUAL = "individual"
    GLOBAL = "global"


class FitSegment(Enum):
    """Enum to track which segment of the data has been fitted."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    BOTH = "both"


class ParamConfig(BaseModel):
    """Configuration for a single fitting parameter."""

    lower_bound: float | None = None
    upper_bound: float | None = None
    initial_value: float | None = None
    global_param: bool = False
    fixed: bool = False

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        """
        Validate that lower bounds are not greater than upper bounds.

        Returns
        -------
        Self
            The validated ParamConfig instance.
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValidationError("lower bound cannot be greater than upper bound")
        return self


class FitConfig(BaseModel):
    """Configuration for fitting operations on oscillation data."""

    params_config: dict[str, ParamConfig] = {}
    auto_calculate_and_fix_f_offset: bool
    f_offset_auto_detect_distance_range_um: tuple[float, float] = (10, 12)
    model_name: str = "fit"
    segment: FitSegment

    @model_validator(mode="after")
    def check_f_offset_not_global_if_auto_calculating(self) -> Self:
        """
        Validate that f_offset is not a global parameter if auto_calculate_and_fix_f_offset is True.

        Returns
        -------
        Self
            The validated FitConfig instance.
        """
        if self.auto_calculate_and_fix_f_offset:
            f_offset_config = self.params_config.get("f_offset")
            if f_offset_config is not None and f_offset_config.global_param:
                raise ValidationError(
                    "f_offset cannot be a global parameter if auto_calculate_and_fix_f_offset is True"
                )
        return self
