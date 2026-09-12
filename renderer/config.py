"""Renderer-local configuration for lighting and screen-space shadows."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LightingConfig:
    """Linear-light controls for ambient, diffuse, and Blinn-Phong specular."""

    ambient_strength: float = 0.15
    specular_strength: float = 0.20
    shininess: float = 48.0
    attenuation_k: float = 0.6

    def __post_init__(self) -> None:
        values = {
            "ambient_strength": self.ambient_strength,
            "specular_strength": self.specular_strength,
            "shininess": self.shininess,
            "attenuation_k": self.attenuation_k,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.shininess <= 0.0:
            raise ValueError("shininess must be > 0")


@dataclass(frozen=True)
class ShadowConfig:
    """Controls for the first fixed-step screen-space point-light shadow pass."""

    shadow_enabled: bool = True
    shadow_resolution_scale: float = 0.5
    shadow_steps: int = 12
    shadow_bias_m: float = 0.015
    shadow_thickness_m: float = 0.15
    ray_start_offset: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.shadow_enabled, bool):
            raise TypeError("shadow_enabled must be a bool")
        if not math.isfinite(self.shadow_resolution_scale) or not 0.0 < self.shadow_resolution_scale <= 1.0:
            raise ValueError("shadow_resolution_scale must be finite and in (0, 1]")
        if isinstance(self.shadow_steps, bool) or not isinstance(self.shadow_steps, int):
            raise TypeError("shadow_steps must be an integer")
        if not 1 <= self.shadow_steps <= 64:
            raise ValueError("shadow_steps must be between 1 and 64")
        for name, value in (
            ("shadow_bias_m", self.shadow_bias_m),
            ("shadow_thickness_m", self.shadow_thickness_m),
            ("ray_start_offset", self.ray_start_offset),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.shadow_thickness_m <= self.shadow_bias_m:
            raise ValueError("shadow_thickness_m must be greater than shadow_bias_m")
