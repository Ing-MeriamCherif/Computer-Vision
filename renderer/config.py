"""Renderer-local configuration for the single-light Level 02 pass."""

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
