"""Renderer-local configuration for lighting and screen-space shadows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class ShadowQualityProfile(str, Enum):
    BASELINE = "baseline"
    SAFE = "safe"
    BALANCED = "balanced"
    HIGH = "high"


class SecondaryShadowMode(str, Enum):
    """Quality fallback applied only to the secondary point light's shadow."""

    BALANCED = "balanced"
    SAFE = "safe"
    OFF = "off"


class VolumetricQualityProfile(str, Enum):
    SAFE = "safe"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass(frozen=True)
class VolumetricConfig:
    """Low-resolution camera-ray scattering controls; disabled by default."""

    volumetric_enabled: bool = False
    volumetric_resolution_scale: float = 0.25
    volumetric_samples: int = 12
    volumetric_density: float = 0.8
    volumetric_intensity: float = 0.15
    volumetric_decay: float = 0.95

    @classmethod
    def for_profile(cls, profile: VolumetricQualityProfile | str) -> "VolumetricConfig":
        try:
            selected = profile if isinstance(profile, VolumetricQualityProfile) else VolumetricQualityProfile(profile.lower())
        except (AttributeError, ValueError) as exc:
            choices = ", ".join(item.value for item in VolumetricQualityProfile)
            raise ValueError(f"volumetric profile must be one of: {choices}") from exc

        profiles = {
            VolumetricQualityProfile.SAFE: {"volumetric_resolution_scale": 0.25, "volumetric_samples": 8},
            VolumetricQualityProfile.BALANCED: {"volumetric_resolution_scale": 0.25, "volumetric_samples": 12},
            VolumetricQualityProfile.HIGH: {"volumetric_resolution_scale": 0.5, "volumetric_samples": 20},
        }
        return cls(**profiles[selected])

    def __post_init__(self) -> None:
        if not isinstance(self.volumetric_enabled, bool):
            raise TypeError("volumetric_enabled must be a bool")
        if (
            not math.isfinite(self.volumetric_resolution_scale)
            or not 0.0 < self.volumetric_resolution_scale <= 1.0
        ):
            raise ValueError("volumetric_resolution_scale must be finite and in (0, 1]")
        if isinstance(self.volumetric_samples, bool) or not isinstance(self.volumetric_samples, int):
            raise TypeError("volumetric_samples must be an integer")
        if not 1 <= self.volumetric_samples <= 24:
            raise ValueError("volumetric_samples must be between 1 and 24")
        for name, value in (
            ("volumetric_density", self.volumetric_density),
            ("volumetric_intensity", self.volumetric_intensity),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not math.isfinite(self.volumetric_decay) or not 0.0 <= self.volumetric_decay <= 1.0:
            raise ValueError("volumetric_decay must be finite and in [0, 1]")


@dataclass(frozen=True)
class LightOrbConfig:
    """Controls the optional perspective-sized, depth-tested light orbs."""

    light_orb_enabled: bool = False
    light_orb_radius_m: float = 0.012
    light_orb_intensity: float = 1.35
    light_orb_halo_strength: float = 0.16
    light_orb_occlusion_bias_m: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.light_orb_enabled, bool):
            raise TypeError("light_orb_enabled must be a bool")
        if not math.isfinite(self.light_orb_radius_m) or self.light_orb_radius_m <= 0.0:
            raise ValueError("light_orb_radius_m must be finite and > 0")
        for name, value in (
            ("light_orb_intensity", self.light_orb_intensity),
            ("light_orb_halo_strength", self.light_orb_halo_strength),
            ("light_orb_occlusion_bias_m", self.light_orb_occlusion_bias_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True)
class ShadowConfig:
    """Controls for fixed-step screen-space shadows and edge-aware filtering."""

    shadow_enabled: bool = True
    shadow_resolution_scale: float = 0.5
    shadow_steps: int = 12
    shadow_bias_m: float = 0.015
    shadow_thickness_m: float = 0.15
    ray_start_offset: float = 0.01
    shadow_softening_enabled: bool = False
    shadow_soft_samples: int = 4
    shadow_soft_radius: float = 1.25
    shadow_edge_aware_upsampling: bool = False
    depth_edge_threshold_m: float = 0.20

    @classmethod
    def for_profile(cls, profile: ShadowQualityProfile | str) -> "ShadowConfig":
        """Build one named quality preset; BASELINE preserves the P6 hard mask."""
        try:
            selected = profile if isinstance(profile, ShadowQualityProfile) else ShadowQualityProfile(profile.lower())
        except (AttributeError, ValueError) as exc:
            names = ", ".join(item.value for item in ShadowQualityProfile)
            raise ValueError(f"quality profile must be one of: {names}") from exc

        presets = {
            ShadowQualityProfile.BASELINE: {
                "shadow_resolution_scale": 0.5,
                "shadow_steps": 12,
                "shadow_softening_enabled": False,
                "shadow_soft_samples": 4,
                "shadow_soft_radius": 1.25,
                "shadow_edge_aware_upsampling": False,
            },
            ShadowQualityProfile.SAFE: {
                "shadow_resolution_scale": 0.5,
                "shadow_steps": 8,
                "shadow_softening_enabled": False,
                "shadow_soft_samples": 4,
                "shadow_soft_radius": 0.75,
                "shadow_edge_aware_upsampling": True,
            },
            ShadowQualityProfile.BALANCED: {
                "shadow_resolution_scale": 0.5,
                # 16 keeps this fixed-step marcher from intermittently skipping
                # the synthetic occluder during camera-space light-Z movement.
                "shadow_steps": 16,
                "shadow_softening_enabled": True,
                "shadow_soft_samples": 4,
                "shadow_soft_radius": 1.25,
                "shadow_edge_aware_upsampling": True,
            },
            ShadowQualityProfile.HIGH: {
                "shadow_resolution_scale": 0.5,
                "shadow_steps": 20,
                "shadow_softening_enabled": True,
                "shadow_soft_samples": 8,
                "shadow_soft_radius": 2.0,
                "shadow_edge_aware_upsampling": True,
            },
        }
        return cls(**presets[selected])

    def __post_init__(self) -> None:
        if not isinstance(self.shadow_enabled, bool):
            raise TypeError("shadow_enabled must be a bool")
        if not isinstance(self.shadow_softening_enabled, bool):
            raise TypeError("shadow_softening_enabled must be a bool")
        if not isinstance(self.shadow_edge_aware_upsampling, bool):
            raise TypeError("shadow_edge_aware_upsampling must be a bool")
        if not math.isfinite(self.shadow_resolution_scale) or not 0.0 < self.shadow_resolution_scale <= 1.0:
            raise ValueError("shadow_resolution_scale must be finite and in (0, 1]")
        if isinstance(self.shadow_steps, bool) or not isinstance(self.shadow_steps, int):
            raise TypeError("shadow_steps must be an integer")
        if not 1 <= self.shadow_steps <= 64:
            raise ValueError("shadow_steps must be between 1 and 64")
        if isinstance(self.shadow_soft_samples, bool) or not isinstance(self.shadow_soft_samples, int):
            raise TypeError("shadow_soft_samples must be an integer")
        if self.shadow_soft_samples not in (4, 8):
            raise ValueError("shadow_soft_samples must be 4 or 8")
        for name, value in (
            ("shadow_bias_m", self.shadow_bias_m),
            ("shadow_thickness_m", self.shadow_thickness_m),
            ("ray_start_offset", self.ray_start_offset),
            ("shadow_soft_radius", self.shadow_soft_radius),
            ("depth_edge_threshold_m", self.depth_edge_threshold_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.shadow_thickness_m <= self.shadow_bias_m:
            raise ValueError("shadow_thickness_m must be greater than shadow_bias_m")
        if self.depth_edge_threshold_m <= 0.0:
            raise ValueError("depth_edge_threshold_m must be > 0")
        if self.shadow_softening_enabled and self.shadow_soft_radius <= 0.0:
            raise ValueError("shadow_soft_radius must be > 0 when softening is enabled")
