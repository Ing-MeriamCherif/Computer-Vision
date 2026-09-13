"""OpenGL 3.3 screen-space relighting for P123 GeometryState frames."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
import threading
from typing import Any

import numpy as np

from .lighting import LightState, active_lights
from .state import GeometryState


@dataclass(frozen=True, slots=True)
class LightingQuality:
    volume_divisor: int
    shadow_rays: int
    shadow_steps: int
    volume_steps: int
    volume_shadow_steps: int
    history_weight: float


QUALITY_PROFILES = {
    "low": LightingQuality(6, 2, 4, 4, 3, 0.0),
    "balanced": LightingQuality(4, 4, 6, 6, 4, 0.18),
    "high": LightingQuality(3, 6, 9, 10, 6, 0.20),
}

_VERTEX_SHADER = r"""#version 330 core
out vec2 vUv;
void main() {
    vec2 p = gl_VertexID == 0 ? vec2(-1.0, -1.0) :
             gl_VertexID == 1 ? vec2( 3.0, -1.0) : vec2(-1.0, 3.0);
    vUv = (p + 1.0) * 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_SURFACE_SHADER = r"""#version 330 core
in vec2 vUv;
layout(location=0) out vec4 oSurface;
layout(location=1) out float oShadow;
layout(location=2) out float oDepth;
uniform sampler2D uRgb;
uniform sampler2D uDepth;
uniform sampler2D uNormal;
uniform sampler2D uConfidence;
uniform sampler2D uValid;
uniform sampler2D uPrevShadow;
uniform sampler2D uPrevDepth;
uniform sampler2D uRtVisibility;
uniform vec2 uResolution;
uniform vec4 uCamera; // fx, fy, cx, cy
uniform int uLightCount;
uniform vec3 uLightPosition[2];
uniform vec3 uLightColor[2];
uniform vec4 uLightPower[2]; // effective intensity, confidence, range, source radius
uniform vec4 uLightBeam[2]; // normalized palm direction xyz, directionality
uniform vec2 uLightCone[2]; // outer and inner cosine
uniform float uLightShadowBias[2];
uniform int uShadowRays;
uniform int uShadowSteps;
uniform int uHistoryAllowed;
uniform int uUseRtShadow;
uniform float uHistoryWeight;
uniform float uAmbient;
uniform float uDiffuseStrength;
uniform float uSpecularStrength;
uniform float uShininess;
uniform float uDirectGain;
uniform int uShadowsEnabled;

vec3 toLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

float rayVisibility(vec3 p, vec3 n, vec3 lightPos, float radius, int lightIndex) {
    const vec2 offsets[6] = vec2[6](
        vec2(-0.65,-0.65), vec2(0.65,-0.65), vec2(-0.65,0.65),
        vec2(0.65,0.65), vec2(0.0,-0.75), vec2(0.0,0.75));
    float visibility = 0.0;
    for (int rayIndex=0; rayIndex<6; ++rayIndex) {
        if (rayIndex >= uShadowRays) break;
        vec3 emitter = lightPos + vec3(offsets[rayIndex] * radius, 0.0);
        vec3 start = p + n * max(uLightShadowBias[lightIndex], 0.0005);
        float rayVisible = 1.0;
        for (int stepIndex=0; stepIndex<10; ++stepIndex) {
            if (stepIndex >= uShadowSteps) break;
            float t = stepIndex == uShadowSteps - 1 ? 0.98 :
                mix(0.04, 0.82, (float(stepIndex) + 0.5) / float(max(uShadowSteps, 1)));
            vec3 s = mix(start, emitter, t);
            if (s.z <= 1e-4) continue;
            vec2 pixel = vec2(uCamera.x * s.x / s.z + uCamera.z,
                              uCamera.y * s.y / s.z + uCamera.w);
            vec2 uv = pixel / uResolution;
            if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
            float sceneZ = texture(uDepth, uv).r;
            float bias = max(0.005, 0.012 * s.z);
            if (sceneZ > min(start.z, emitter.z) + bias && sceneZ < s.z - bias) {
                rayVisible = 0.20;
                break;
            }
        }
        visibility += rayVisible;
    }
    return visibility / float(max(uShadowRays, 1));
}

float rtVisibility(vec2 uv, int lightIndex) {
    vec2 visibility = texture(uRtVisibility, uv).rg;
    return lightIndex == 0 ? visibility.r : visibility.g;
}

void main() {
    vec3 source = texture(uRgb, vUv).rgb;
    float z = texture(uDepth, vUv).r;
    float valid = texture(uValid, vUv).r;
    vec3 base = toLinear(source);
    if (valid < 0.5 || z <= 1e-5) {
        oSurface = vec4(base, 1.0);
        oShadow = 1.0;
        oDepth = 0.0;
        return;
    }
    vec2 pixel = vUv * uResolution - vec2(0.5);
    vec3 p = vec3((pixel.x - uCamera.z) * z / uCamera.x,
                  (pixel.y - uCamera.w) * z / uCamera.y, z);
    float normalLength = length(texture(uNormal, vUv).xyz);
    if (!(normalLength > 1e-4)) { oSurface=vec4(base,1.0); oShadow=1.0; oDepth=z; return; }
    vec3 n = texture(uNormal, vUv).xyz / normalLength;
    float conf = clamp(texture(uConfidence, vUv).r, 0.0, 1.0);
    float normalConfidence = smoothstep(0.05, 0.45, length(texture(uNormal, vUv).xyz));
    vec3 diffuseTerm = vec3(0.0);
    vec3 specularTerm = vec3(0.0);
    float visibilityMean = 1.0;
    float visibilitySum = 0.0;
    for (int i=0; i<2; ++i) {
        if (i >= uLightCount) break;
        vec3 delta = uLightPosition[i] - p;
        float distanceToLight = max(length(delta), 1e-5);
        vec3 l = delta / distanceToLight;
        vec3 v = normalize(-p);
        vec3 h = normalize(l + v);
        float diffuse = max(dot(n, l), 0.0);
        float specular = diffuse > 0.0 ? pow(max(dot(n, h), 0.0), uShininess) : 0.0;
        float rangeM = max(uLightPower[i].z, 0.01);
        float attenuation = uLightPower[i].x / (1.0 + pow(distanceToLight / rangeM, 2.0));
        float beamCosine = dot(uLightBeam[i].xyz, -l);
        float beamLobe = smoothstep(uLightCone[i].x, uLightCone[i].y, beamCosine);
        attenuation *= mix(1.0, beamLobe, clamp(uLightBeam[i].w, 0.0, 1.0));
        float visibility = uShadowsEnabled == 0 ? 1.0 : (uUseRtShadow != 0
            ? rtVisibility(vUv, i)
            : rayVisibility(p, n, uLightPosition[i], uLightPower[i].w, i));
        // The stored visibility texture is shared; only a single emitter can
        // safely reuse it. Multi-light frames keep independent current rays.
        if (uLightCount == 1 && uHistoryAllowed != 0 && uHistoryWeight > 0.0) {
            float previousZ = texture(uPrevDepth, vUv).r;
            float previousVisibility = texture(uPrevShadow, vUv).r;
            if (previousZ > 1e-5 && abs(previousZ - z) < max(0.018, 0.025 * z)) {
                visibility = mix(visibility, previousVisibility, uHistoryWeight);
            }
        }
        visibilitySum += visibility;
        float directScale = attenuation * visibility * uLightPower[i].y * conf * normalConfidence;
        diffuseTerm += base * uLightColor[i] * diffuse * uDiffuseStrength * directScale * uDirectGain;
        specularTerm += uLightColor[i] * specular * uSpecularStrength * directScale * uDirectGain;
    }
    if (uLightCount > 0) visibilityMean = visibilitySum / float(uLightCount);
    float ambient = max(uAmbient, 0.05);
    vec3 ambientTerm = base * ambient;
    vec3 shaded = (ambientTerm + diffuseTerm + specularTerm) / ambient;
    oSurface = vec4(shaded, 1.0);
    oShadow = visibilityMean;
    oDepth = z;
}
"""

_VOLUME_SHADER = r"""#version 330 core
in vec2 vUv;
layout(location=0) out vec4 oVolume;
uniform sampler2D uDepth;
uniform sampler2D uValid;
uniform sampler2D uPrevVolume;
uniform sampler2D uPrevDepth;
uniform vec2 uResolution;
uniform vec2 uVolumeResolution;
uniform vec4 uCamera;
uniform int uLightCount;
uniform vec3 uLightPosition[2];
uniform vec3 uLightColor[2];
uniform vec4 uLightPower[2];
uniform vec4 uLightBeam[2]; // normalized palm direction xyz, directionality
uniform vec2 uLightCone[2]; // outer and inner cosine
uniform float uLightShadowBias[2];
uniform int uVolumeSteps;
uniform int uVolShadowSteps;
uniform int uHistoryAllowed;
uniform float uHistoryWeight;
uniform float uDensity;

float lightVisibility(vec3 samplePos, vec3 lightPos, int lightIndex) {
    float visible = 1.0;
    vec3 ray = lightPos - samplePos;
    vec3 start = samplePos + normalize(ray) * max(uLightShadowBias[lightIndex], 0.0005);
    for (int i=0; i<6; ++i) {
        if (i >= uVolShadowSteps) break;
        float t = i == uVolShadowSteps - 1 ? 0.98 :
            mix(0.04, 0.82, (float(i) + 0.5) / float(max(uVolShadowSteps, 1)));
        vec3 s = mix(start, lightPos, t);
        if (s.z <= 1e-4) continue;
        vec2 pixel = vec2(uCamera.x * s.x / s.z + uCamera.z,
                          uCamera.y * s.y / s.z + uCamera.w);
        vec2 uv = pixel / uResolution;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
        float sceneZ = texture(uDepth, uv).r;
        float bias = max(0.008, 0.014 * s.z);
        if (sceneZ > min(start.z, lightPos.z) + bias && sceneZ < s.z - bias) {
            visible = 0.18;
            break;
        }
    }
    return visible;
}

void main() {
    float surfaceZ = texture(uDepth, vUv).r;
    float valid = texture(uValid, vUv).r;
    float maxZ = valid > 0.5 && surfaceZ > 1e-5 ? surfaceZ : 3.0;
    vec2 pixel = vUv * uResolution - vec2(0.5);
    vec3 ray = vec3((pixel.x-uCamera.z)/uCamera.x, (pixel.y-uCamera.w)/uCamera.y, 1.0);
    vec3 haze = vec3(0.0);
    float nearZ = 0.15;
    float farZ = max(maxZ, 0.2);
    float stepLength = ((farZ - nearZ) / float(max(uVolumeSteps, 1))) * length(ray);
    for (int sampleIndex=0; sampleIndex<12; ++sampleIndex) {
        if (sampleIndex >= uVolumeSteps) break;
        float sampleZ = nearZ + (float(sampleIndex) + 0.5) * (farZ - nearZ) / float(max(uVolumeSteps, 1));
        vec3 s = ray * sampleZ;
        if (valid > 0.5 && surfaceZ > 1e-5 && sampleZ > surfaceZ - max(0.004, 0.008 * surfaceZ)) continue;
        for (int lightIndex=0; lightIndex<2; ++lightIndex) {
            if (lightIndex >= uLightCount) break;
            vec3 delta = uLightPosition[lightIndex] - s;
            float distanceSq = max(dot(delta, delta), 0.0025);
            float rangeM = max(uLightPower[lightIndex].z * 0.78, 0.01);
            float rangeWeight = exp(-0.5 * distanceSq / (rangeM * rangeM));
            float beamCosine = dot(uLightBeam[lightIndex].xyz, normalize(-delta));
            float beamLobe = smoothstep(uLightCone[lightIndex].x, uLightCone[lightIndex].y, beamCosine);
            rangeWeight *= mix(1.0, beamLobe, clamp(uLightBeam[lightIndex].w, 0.0, 1.0));
            float visibility = lightVisibility(s, uLightPosition[lightIndex], lightIndex);
            float scatter = uDensity * uLightPower[lightIndex].x * uLightPower[lightIndex].y;
            haze += uLightColor[lightIndex] * rangeWeight * visibility * scatter * stepLength;
        }
    }
    if (uHistoryAllowed != 0 && uHistoryWeight > 0.0) {
        float currentZ = valid > 0.5 ? surfaceZ : 0.0;
        float previousZComparable = texture(uPrevDepth, vUv).r;
        bool sameGeometry = (currentZ <= 1e-5 && previousZComparable <= 1e-5) ||
            (currentZ > 1e-5 && abs(previousZComparable - currentZ) < max(0.02, 0.03 * currentZ));
        if (sameGeometry) haze = mix(haze, texture(uPrevVolume, vUv).rgb, uHistoryWeight);
    }
    oVolume = vec4(haze, 1.0);
}
"""

_COMPOSITE_SHADER = r"""#version 330 core
in vec2 vUv;
layout(location=0) out vec4 oColor;
uniform sampler2D uSurface;
uniform sampler2D uVolume;
uniform sampler2D uDepth;
uniform vec2 uResolution;
uniform vec2 uVolumeResolution;
uniform vec4 uCamera;
uniform int uLightCount;
uniform vec3 uLightPosition[2];
uniform vec3 uLightColor[2];
uniform vec4 uLightPower[2];
uniform float uLightVisualRadius[2];
uniform vec2 uLightVisualMeta[2]; // orb visibility, palm-attached

vec3 toSrgb(vec3 c) {
    c = max(c, vec3(0.0));
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055, step(vec3(0.0031308), c));
}

void main() {
    vec3 color = texture(uSurface, vUv).rgb + texture(uVolume, vUv).rgb;
    vec2 pixel = vUv * uResolution - vec2(0.5);
    for (int i=0; i<2; ++i) {
        if (i >= uLightCount) break;
        vec3 p = uLightPosition[i];
        if (p.z <= 1e-4) continue;
        float orbVisibility = clamp(uLightVisualMeta[i].x, 0.0, 1.0);
        if (orbVisibility <= 0.001) continue;
        bool palmAttached = uLightVisualMeta[i].y > 0.5;
        vec2 center = vec2(uCamera.x * p.x / p.z + uCamera.z,
                           uCamera.y * p.y / p.z + uCamera.w);
        vec2 centerUv = center / uResolution;
        float centerSceneZ = texture(uDepth, clamp(centerUv, vec2(0.0), vec2(1.0))).r;
        float centerBias = palmAttached ? max(0.006, 0.01 * p.z) : max(0.025, 0.04 * p.z);
        bool sourceOccluded = centerSceneZ > 1e-5 && centerSceneZ + centerBias < p.z;
        if (palmAttached && sourceOccluded) continue;
        float radius = clamp(uCamera.x * max(uLightVisualRadius[i], 0.004) / p.z, 6.0, 32.0);
        float d = length(pixel - center);
        float normalized = d / radius;
        float outer = exp(-0.5 * pow(d / max(radius * 1.35, 1.0), 2.0));
        float inner = exp(-0.5 * pow(d / max(radius * 0.88, 1.0), 2.0));
        float orbTrackingPower = palmAttached ? 1.0 : uLightPower[i].y;
        float power = clamp(uLightPower[i].x * orbTrackingPower, 0.0, 2.0);
        float sceneZ = texture(uDepth, vUv).r;
        float depthBias = palmAttached ? max(0.006, 0.01 * p.z) : max(0.025, 0.04 * p.z);
        bool depthOccluded = sceneZ > 1e-5 && sceneZ + depthBias < p.z;
        if (palmAttached && depthOccluded) continue;
        float haloVisibility = depthOccluded ? 0.28 : 1.0;
        color += uLightColor[i] * max(outer-inner, 0.0) * 0.008 * power * haloVisibility * orbVisibility;
        color += uLightColor[i] * inner * 0.025 * power * mix(0.55, 1.0, haloVisibility) * orbVisibility;
        if (normalized < 1.0) {
            float sphereZ = sqrt(max(1.0 - normalized * normalized, 0.0));
            vec2 orbXY = (pixel - center) / radius;
            float highlight = exp(-0.5 * dot((orbXY - vec2(-0.24, -0.30)) / 0.12,
                                              (orbXY - vec2(-0.24, -0.30)) / 0.12));
            float whiteMix = clamp(0.42 + 0.36 * sphereZ + 0.55 * highlight, 0.0, 1.0);
            vec3 ballColor = mix(uLightColor[i], vec3(1.0, 0.98, 0.93), whiteMix);
            float rim = pow(1.0 - sphereZ, 2.0) * 0.22;
            ballColor = mix(ballColor, uLightColor[i], rim);
            float edge = 1.0 - smoothstep(0.88, 1.0, normalized);
            color = mix(color, ballColor, edge * 0.90 * orbVisibility);
        }
    }
    oColor = vec4(clamp(toSrgb(color), 0.0, 1.0), 1.0);
}
"""


class GPURelightRenderer:
    """Persistent GL 3.3 renderer; outputs NumPy RGB for the existing P123 UI."""

    def __init__(self, quality: str = "balanced", *, share_window=None) -> None:
        self._glfw = None
        self._gl = None
        self._window = None
        self._share_window = share_window
        self._glfw_acquired = False
        self._width = 0
        self._height = 0
        self._volume_width = 0
        self._volume_height = 0
        self._programs: dict[str, int] = {}
        self._textures: dict[str, int] = {}
        self._framebuffers: dict[str, int] = {}
        self._vao = 0
        self._history_index = 0
        self._history_key: tuple[Any, ...] | None = None
        self._history_time = 0.0
        self._render_lock = threading.Lock()
        self._rt_renderer = None
        self._rt_init_error = "not initialized"
        self._rt_mode = os.environ.get("NRW_RAY_BACKEND", "auto").strip().lower()
        try:
            self._rt_width = max(32, int(os.environ.get("NRW_RT_WIDTH", "96")))
            self._rt_height = max(18, int(os.environ.get("NRW_RT_HEIGHT", "54")))
        except ValueError:
            self._rt_width, self._rt_height = 96, 54
        self.initialized = False
        self.gl_version = "unknown"
        self.last_stats: dict[str, float | str] = {}
        self.set_quality(quality)
        self._initialize()

    def set_quality(self, quality: str) -> None:
        value = str(quality).lower()
        if value not in QUALITY_PROFILES:
            raise ValueError(f"unknown lighting quality '{quality}'")
        if not getattr(self, "initialized", False):
            self.quality_name = value
            self.quality = QUALITY_PROFILES[value]
            return
        with self._render_lock:
            previous = self.quality
            self.quality_name = value
            self.quality = QUALITY_PROFILES[value]
            self._history_key = None
            self._history_time = 0.0
            if previous.volume_divisor != self.quality.volume_divisor and self._width > 0 and self._height > 0:
                self._glfw.make_context_current(self._window)
                try:
                    self.resize(self._width, self._height)
                finally:
                    self._glfw.make_context_current(None)

    def _initialize(self) -> None:
        import glfw
        import OpenGL.GL as gl

        self._glfw, self._gl = glfw, gl
        from .glfw_support import acquire

        if not acquire(glfw):
            raise RuntimeError("GLFW initialization failed")
        self._glfw_acquired = True
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
            self._window = glfw.create_window(32, 32, "P123 GPU relight", None, self._share_window)
            if not self._window:
                raise RuntimeError("could not create an OpenGL 3.3 core context")
            glfw.make_context_current(self._window)
            self.gl_version = gl.glGetString(gl.GL_VERSION).decode("ascii", "replace")
            if not self._supports_gl33(self.gl_version):
                raise RuntimeError(f"OpenGL 3.3+ required, got {self.gl_version}")
            for name, shader in (
                ("surface", _SURFACE_SHADER),
                ("volume", _VOLUME_SHADER),
                ("composite", _COMPOSITE_SHADER),
            ):
                self._programs[name] = self._link_program(_VERTEX_SHADER, shader)
            self._vao = int(gl.glGenVertexArrays(1))
            gl.glBindVertexArray(self._vao)
            if self._rt_mode in {"auto", "optix", "rtx"}:
                try:
                    from .optix_relighting import create_optix_renderer

                    self._rt_renderer, self._rt_init_error = create_optix_renderer(self._rt_width, self._rt_height)
                except Exception as exc:
                    self._rt_init_error = f"{type(exc).__name__}: {exc}"
            self.initialized = True
            glfw.make_context_current(None)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _supports_gl33(version: str) -> bool:
        try:
            major, minor = (int(part) for part in version.split()[0].split(".")[:2])
            return (major, minor) >= (3, 3)
        except (ValueError, IndexError):
            return False

    def _link_program(self, vertex_source: str, fragment_source: str) -> int:
        gl = self._gl
        shaders: list[int] = []
        program = 0
        try:
            for kind, source in ((gl.GL_VERTEX_SHADER, vertex_source), (gl.GL_FRAGMENT_SHADER, fragment_source)):
                shader = gl.glCreateShader(kind)
                shaders.append(shader)
                gl.glShaderSource(shader, source)
                gl.glCompileShader(shader)
                if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                    message = gl.glGetShaderInfoLog(shader).decode("utf-8", "replace")
                    raise RuntimeError(f"GLSL compile failed: {message}")
            program = gl.glCreateProgram()
            for shader in shaders:
                gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)
            if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
                message = gl.glGetProgramInfoLog(program).decode("utf-8", "replace")
                raise RuntimeError(f"GLSL link failed: {message}")
            return int(program)
        except Exception:
            if program:
                gl.glDeleteProgram(program)
            raise
        finally:
            for shader in shaders:
                gl.glDeleteShader(shader)

    def _texture(self, name: str, width: int, height: int, internal: int, fmt: int, typ: int, *, filtering: int) -> int:
        gl = self._gl
        texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filtering)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filtering)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, typ, None)
        self._textures[name] = texture
        return texture

    def _delete_frame_resources(self) -> None:
        gl = self._gl
        if self._textures:
            textures = list(self._textures.values())
            gl.glDeleteTextures(len(textures), textures)
        if self._framebuffers:
            framebuffers = list(self._framebuffers.values())
            gl.glDeleteFramebuffers(len(framebuffers), framebuffers)
        self._textures.clear()
        self._framebuffers.clear()

    def resize(self, width: int, height: int) -> None:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        volume_width = max(1, (width + self.quality.volume_divisor - 1) // self.quality.volume_divisor)
        volume_height = max(1, (height + self.quality.volume_divisor - 1) // self.quality.volume_divisor)
        if (width, height, volume_width, volume_height) == (self._width, self._height, self._volume_width, self._volume_height):
            return
        self._glfw.make_context_current(self._window)
        self._delete_frame_resources()
        self._width, self._height = width, height
        self._volume_width, self._volume_height = volume_width, volume_height
        gl = self._gl
        nearest, linear = gl.GL_NEAREST, gl.GL_LINEAR
        self._texture("rgb", width, height, gl.GL_RGB8, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, filtering=linear)
        self._texture("depth", width, height, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, filtering=nearest)
        self._texture("normal", width, height, gl.GL_RGB16F, gl.GL_RGB, gl.GL_FLOAT, filtering=nearest)
        self._texture("confidence", width, height, gl.GL_R16F, gl.GL_RED, gl.GL_FLOAT, filtering=nearest)
        self._texture("valid", width, height, gl.GL_R8, gl.GL_RED, gl.GL_UNSIGNED_BYTE, filtering=nearest)
        self._texture("surface", width, height, gl.GL_RGBA16F, gl.GL_RGBA, gl.GL_FLOAT, filtering=linear)
        self._texture("output", width, height, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, filtering=linear)
        self._texture("rt_visibility", self._rt_width, self._rt_height,
                      gl.GL_RG32F, gl.GL_RG, gl.GL_FLOAT, filtering=linear)
        for i in range(2):
            self._texture(f"shadow{i}", width, height, gl.GL_R16F, gl.GL_RED, gl.GL_FLOAT, filtering=linear)
            self._texture(f"history_depth{i}", width, height, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, filtering=nearest)
            self._texture(f"volume{i}", volume_width, volume_height, gl.GL_RGBA16F, gl.GL_RGBA, gl.GL_FLOAT, filtering=linear)
        self._framebuffers["surface"] = int(gl.glGenFramebuffers(1))
        self._framebuffers["volume"] = int(gl.glGenFramebuffers(1))
        self._framebuffers["output"] = int(gl.glGenFramebuffers(1))
        self._history_index = 0
        self._history_key = None
        self._history_time = 0.0

    def _attach_surface_targets(self, write_index: int) -> None:
        gl = self._gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._framebuffers["surface"])
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._textures["surface"], 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, self._textures[f"shadow{write_index}"], 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, self._textures[f"history_depth{write_index}"], 0)
        gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])
        self._check_framebuffer("surface")

    def _check_framebuffer(self, name: str) -> None:
        status = self._gl.glCheckFramebufferStatus(self._gl.GL_FRAMEBUFFER)
        if status != self._gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"{name} framebuffer incomplete: 0x{int(status):04x}")

    def _upload(self, texture: str, data: np.ndarray, fmt: int, typ: int) -> None:
        gl = self._gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures[texture])
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._width, self._height, fmt, typ, data)

    @staticmethod
    def _uniform(program: int, name: str) -> int:
        return int(GPURelightRenderer._current_gl.glGetUniformLocation(program, name))

    @property
    def _current_gl(self):
        return self._gl

    def _set_lights(self, program: int, lights: list[LightState]) -> None:
        gl = self._gl
        positions = np.zeros((2, 3), dtype=np.float32)
        colors = np.zeros((2, 3), dtype=np.float32)
        powers = np.zeros((2, 4), dtype=np.float32)
        beams = np.zeros((2, 4), dtype=np.float32)
        cones = np.zeros((2, 2), dtype=np.float32)
        visual_meta = np.zeros((2, 2), dtype=np.float32)
        shadow_bias = np.full(2, 0.002, dtype=np.float32)
        visual_radii = np.zeros(2, dtype=np.float32)
        for i, light in enumerate(lights[:2]):
            positions[i] = np.asarray(light.position_camera, dtype=np.float32)
            colors[i] = np.clip(np.asarray(light.color_rgb, dtype=np.float32), 0.0, 1.0)
            powers[i] = (
                max(float(light.effective_intensity), 0.0),
                float(np.clip(light.confidence, 0.0, 1.0)),
                max(float(getattr(light, "range_m", 0.30)), 0.01),
                max(float(getattr(light, "source_radius_m", 0.018)), 0.001),
            )
            normal = getattr(light, "palm_normal_camera", None)
            if light.is_palm_attached and normal is not None:
                direction = np.asarray(normal, dtype=np.float32).reshape(3)
                direction_norm = float(np.linalg.norm(direction))
                if np.isfinite(direction).all() and direction_norm > 1e-6:
                    beams[i, :3] = direction / direction_norm
                    beams[i, 3] = float(np.clip(getattr(light, "directionality", 0.78), 0.0, 1.0))
            inner_deg = float(np.clip(getattr(light, "beam_inner_angle_deg", 32.0), 1.0, 89.0))
            outer_deg = float(np.clip(getattr(light, "beam_outer_angle_deg", 78.0), inner_deg + 1.0, 179.0))
            cones[i] = (np.cos(np.deg2rad(outer_deg)), np.cos(np.deg2rad(inner_deg)))
            visual_meta[i] = (
                float(np.clip(light.orb_visibility, 0.0, 1.0)),
                1.0 if light.is_palm_attached else 0.0,
            )
            shadow_bias[i] = float(np.clip(light.self_intersection_epsilon_m, 0.0001, 0.02))
            visual_radii[i] = max(float(getattr(light, "visual_radius_m", 0.035)), 0.004)
        gl.glUniform1i(gl.glGetUniformLocation(program, "uLightCount"), min(len(lights), 2))
        gl.glUniform3fv(gl.glGetUniformLocation(program, "uLightPosition[0]"), 2, positions)
        gl.glUniform3fv(gl.glGetUniformLocation(program, "uLightColor[0]"), 2, colors)
        gl.glUniform4fv(gl.glGetUniformLocation(program, "uLightPower[0]"), 2, powers)
        beam_location = gl.glGetUniformLocation(program, "uLightBeam[0]")
        if beam_location >= 0:
            gl.glUniform4fv(beam_location, 2, beams)
        cone_location = gl.glGetUniformLocation(program, "uLightCone[0]")
        if cone_location >= 0:
            gl.glUniform2fv(cone_location, 2, cones)
        visual_meta_location = gl.glGetUniformLocation(program, "uLightVisualMeta[0]")
        if visual_meta_location >= 0:
            gl.glUniform2fv(visual_meta_location, 2, visual_meta)
        shadow_bias_location = gl.glGetUniformLocation(program, "uLightShadowBias[0]")
        if shadow_bias_location >= 0:
            gl.glUniform1fv(shadow_bias_location, 2, shadow_bias)
        visual_radius_location = gl.glGetUniformLocation(program, "uLightVisualRadius[0]")
        if visual_radius_location >= 0:
            gl.glUniform1fv(visual_radius_location, 2, visual_radii)

    def _bind_texture(self, unit: int, texture: int, program: int, uniform_name: str) -> None:
        gl = self._gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glUniform1i(gl.glGetUniformLocation(program, uniform_name), unit)

    def _draw(self, program: int, width: int, height: int) -> None:
        gl = self._gl
        gl.glViewport(0, 0, width, height)
        gl.glUseProgram(program)
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def _history_compatible(self, geometry: GeometryState, lights: list[LightState], now: float) -> bool:
        key = (
            self._width,
            self._height,
            geometry.source_frame_id,
            tuple(light.light_id for light in lights),
            tuple(tuple(np.round(light.position_camera, 2)) for light in lights),
        )
        old = self._history_key
        frame_contiguous = False
        if old is not None:
            previous_frame = old[2]
            current_frame = geometry.source_frame_id
            if isinstance(previous_frame, int) and isinstance(current_frame, int):
                frame_contiguous = 0 <= current_frame - previous_frame <= 4
            else:
                frame_contiguous = previous_frame == current_frame
        light_set_same = old is not None and old[3] == key[3]
        positions_stable = old is not None and old[4] == key[4]
        compatible = bool(
            old is not None
            and old[0:2] == key[0:2]
            and frame_contiguous
            and light_set_same
            and positions_stable
            and now - self._history_time < 0.12
        )
        self._history_key = key
        self._history_time = now
        return compatible

    def reset_history(self) -> None:
        """Invalidate temporal shadows and volumes after a camera source change."""
        with self._render_lock:
            self._history_key = None
            self._history_time = 0.0

    def render(
        self,
        rgb: np.ndarray,
        geometry: GeometryState,
        lights: list[LightState] | tuple[LightState, ...],
        *,
        ambient: float = 0.40,
        readback: bool = True,
        stage: str = "full",
        diffuse_strength: float = 0.92,
        specular_strength: float = 0.12,
        shininess: float = 36.0,
        direct_gain: float = 1.0,
    ) -> tuple[np.ndarray | None, dict[str, float | str]]:
        with self._render_lock:
            try:
                return self._render_impl(rgb, geometry, lights, ambient=ambient, readback=readback, stage=stage, diffuse_strength=diffuse_strength, specular_strength=specular_strength, shininess=shininess, direct_gain=direct_gain)
            finally:
                if self._window is not None:
                    self._glfw.make_context_current(None)

    def _render_impl(
        self,
        rgb: np.ndarray,
        geometry: GeometryState,
        lights: list[LightState] | tuple[LightState, ...],
        *,
        ambient: float = 0.40,
        readback: bool = True,
        stage: str = "full",
        diffuse_strength: float = 0.92,
        specular_strength: float = 0.12,
        shininess: float = 36.0,
        direct_gain: float = 1.0,
    ) -> tuple[np.ndarray | None, dict[str, float | str]]:
        if not self.initialized:
            raise RuntimeError("GPU relight renderer is not initialized")
        render_started = time.perf_counter()
        frame = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
        height, width = frame.shape[:2]
        if geometry.depth.shape != (height, width) or geometry.normals is None:
            raise ValueError("GPU relighting requires full-resolution depth and normals matching RGB")
        if (width, height) != (self._width, self._height):
            self.resize(width, height)
        self._glfw.make_context_current(self._window)
        gl = self._gl
        valid = np.asarray(geometry.valid_mask, dtype=bool) & np.isfinite(geometry.depth) & (geometry.depth > 1e-5)
        depth = np.where(valid, geometry.depth, 0.0).astype(np.float32, copy=False)
        normals = np.nan_to_num(np.asarray(geometry.normals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        confidence = geometry.confidence
        if confidence is None:
            confidence = valid.astype(np.float32)
        confidence = np.clip(np.nan_to_num(np.asarray(confidence, dtype=np.float32), nan=0.0), 0.0, 1.0)
        valid_tex = valid.astype(np.uint8) * 255
        self._upload("rgb", frame, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self._upload("depth", depth, gl.GL_RED, gl.GL_FLOAT)
        self._upload("normal", np.ascontiguousarray(normals), gl.GL_RGB, gl.GL_FLOAT)
        self._upload("confidence", np.ascontiguousarray(confidence), gl.GL_RED, gl.GL_FLOAT)
        self._upload("valid", np.ascontiguousarray(valid_tex), gl.GL_RED, gl.GL_UNSIGNED_BYTE)
        active = active_lights(lights)[:2]
        rt_trace_ms = 0.0
        rt_active = self._rt_renderer is not None and self._rt_mode in {"auto", "optix", "rtx"}
        if rt_active:
            try:
                rt_visibility, rt_trace_ms = self._rt_renderer.render(
                    geometry.depth, valid, normals, geometry.camera, active
                )
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures["rt_visibility"])
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._rt_width, self._rt_height,
                                   gl.GL_RG, gl.GL_FLOAT, np.ascontiguousarray(rt_visibility))
            except Exception as exc:
                self._rt_init_error = f"{type(exc).__name__}: {exc}"
                self._rt_renderer = None
                rt_active = False
        now = time.monotonic()
        history_ok = self._history_compatible(geometry, active, now)
        previous_index = self._history_index
        write_index = 1 - previous_index
        fx, fy, cx, cy = geometry.camera.fx, geometry.camera.fy, geometry.camera.cx, geometry.camera.cy
        cam = np.asarray([fx, fy, cx, cy], dtype=np.float32)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_DEPTH_TEST)

        surface_program = self._programs["surface"]
        self._attach_surface_targets(write_index)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(surface_program)
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uLightCount"), min(len(active), 2))
        self._set_lights(surface_program, active)
        gl.glUniform2f(gl.glGetUniformLocation(surface_program, "uResolution"), width, height)
        gl.glUniform4fv(gl.glGetUniformLocation(surface_program, "uCamera"), 1, cam)
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uShadowRays"), self.quality.shadow_rays)
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uShadowSteps"), self.quality.shadow_steps)
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uHistoryAllowed"), int(history_ok))
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uUseRtShadow"), int(rt_active))
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uHistoryWeight"), self.quality.history_weight)
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uAmbient"), float(ambient))
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uDiffuseStrength"), float(diffuse_strength))
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uSpecularStrength"), float(specular_strength if stage != "l2_diffuse" else 0.0))
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uShininess"), float(shininess))
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uDirectGain"), float(direct_gain))
        gl.glUniform1i(gl.glGetUniformLocation(surface_program, "uShadowsEnabled"), int(stage in {"full", "l4_shadows"}))
        self._bind_texture(0, self._textures["rgb"], surface_program, "uRgb")
        self._bind_texture(1, self._textures["depth"], surface_program, "uDepth")
        self._bind_texture(2, self._textures["normal"], surface_program, "uNormal")
        self._bind_texture(3, self._textures["confidence"], surface_program, "uConfidence")
        self._bind_texture(4, self._textures["valid"], surface_program, "uValid")
        self._bind_texture(5, self._textures[f"shadow{previous_index}"], surface_program, "uPrevShadow")
        self._bind_texture(6, self._textures[f"history_depth{previous_index}"], surface_program, "uPrevDepth")
        self._bind_texture(7, self._textures["rt_visibility"], surface_program, "uRtVisibility")
        pass_started = time.perf_counter()
        self._draw(surface_program, width, height)
        shadow_submit_ms = (time.perf_counter() - pass_started) * 1000.0

        volume_program = self._programs["volume"]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._framebuffers["volume"])
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._textures[f"volume{write_index}"], 0)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        self._check_framebuffer("volume")
        gl.glUseProgram(volume_program)
        gl.glUniform2f(gl.glGetUniformLocation(volume_program, "uResolution"), width, height)
        gl.glUniform2f(gl.glGetUniformLocation(volume_program, "uVolumeResolution"), self._volume_width, self._volume_height)
        gl.glUniform4fv(gl.glGetUniformLocation(volume_program, "uCamera"), 1, cam)
        self._set_lights(volume_program, active)
        gl.glUniform1i(gl.glGetUniformLocation(volume_program, "uVolumeSteps"), self.quality.volume_steps)
        gl.glUniform1i(gl.glGetUniformLocation(volume_program, "uVolShadowSteps"), self.quality.volume_shadow_steps)
        gl.glUniform1i(gl.glGetUniformLocation(volume_program, "uHistoryAllowed"), int(history_ok))
        gl.glUniform1f(gl.glGetUniformLocation(volume_program, "uHistoryWeight"), self.quality.history_weight)
        gl.glUniform1f(gl.glGetUniformLocation(volume_program, "uDensity"), 0.12)
        self._bind_texture(1, self._textures["depth"], volume_program, "uDepth")
        self._bind_texture(2, self._textures["valid"], volume_program, "uValid")
        self._bind_texture(5, self._textures[f"volume{previous_index}"], volume_program, "uPrevVolume")
        self._bind_texture(6, self._textures[f"history_depth{previous_index}"], volume_program, "uPrevDepth")
        pass_started = time.perf_counter()
        self._draw(volume_program, self._volume_width, self._volume_height)
        volumetric_submit_ms = (time.perf_counter() - pass_started) * 1000.0

        composite_program = self._programs["composite"]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._framebuffers["output"])
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._textures["output"], 0)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        self._check_framebuffer("output")
        gl.glUseProgram(composite_program)
        gl.glUniform2f(gl.glGetUniformLocation(composite_program, "uResolution"), width, height)
        gl.glUniform2f(gl.glGetUniformLocation(composite_program, "uVolumeResolution"), self._volume_width, self._volume_height)
        gl.glUniform4fv(gl.glGetUniformLocation(composite_program, "uCamera"), 1, cam)
        self._set_lights(composite_program, active)
        self._bind_texture(0, self._textures["surface"], composite_program, "uSurface")
        self._bind_texture(1, self._textures[f"volume{write_index}"], composite_program, "uVolume")
        self._bind_texture(2, self._textures["depth"], composite_program, "uDepth")
        pass_started = time.perf_counter()
        self._draw(composite_program, width, height)
        composite_submit_ms = (time.perf_counter() - pass_started) * 1000.0
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        result = None
        if readback:
            output = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            result = np.frombuffer(output, dtype=np.uint8).reshape(height, width, 3).copy()
        else:
            gl.glFlush()
        self._history_index = write_index
        total_ms = (time.perf_counter() - render_started) * 1000.0
        self.last_stats = {
            "renderer": "GPU",
            "ray_backend": "NVIDIA_OPTIX_RT_CORES" if rt_active else "GLSL_SCREEN_SPACE_FALLBACK",
            "rt_trace_ms": float(rt_trace_ms),
            "rt_resolution": f"{self._rt_width}x{self._rt_height}",
            "gpu_render_ms": float(total_ms),
            "shadow_submit_ms": float(shadow_submit_ms),
            "volumetric_submit_ms": float(volumetric_submit_ms),
            "composite_submit_ms": float(composite_submit_ms),
            "lights": float(len(active)),
            "quality": self.quality_name,
            "shadow_quality": f"{self.quality.shadow_rays}x{self.quality.shadow_steps}",
            "volumetric_quality": f"{self.quality.volume_steps}/{self.quality.volume_shadow_steps}",
            "gl_version": self.gl_version,
            "volumetric_resolution": f"{self._volume_width}x{self._volume_height}",
        }
        if not rt_active:
            self.last_stats["ray_backend_note"] = self._rt_init_error
        return result, dict(self.last_stats)

    def render_to_texture(
        self,
        rgb: np.ndarray,
        geometry: GeometryState,
        lights: list[LightState] | tuple[LightState, ...],
        *,
        ambient: float = 0.40,
    ) -> tuple[int, dict[str, float | str]]:
        """Render without CPU readback for a NativeOpenGLWindow sharing this context."""
        self.render(rgb, geometry, lights, ambient=ambient, readback=False)
        return self.output_texture, dict(self.last_stats)

    @property
    def output_texture(self) -> int:
        """Current output texture name; share this context to present it directly."""
        return self._textures.get("output", 0)

    def close(self) -> None:
        gl, glfw = self._gl, self._glfw
        with self._render_lock:
            if gl is not None:
                try:
                    if self._window is not None:
                        glfw.make_context_current(self._window)
                    self._delete_frame_resources()
                    for program in self._programs.values():
                        gl.glDeleteProgram(program)
                    self._programs.clear()
                    if self._vao:
                        gl.glDeleteVertexArrays(1, [self._vao])
                        self._vao = 0
                except Exception as exc:
                    print(f"GPU RELIGHT CLEANUP FAILED: {type(exc).__name__}: {exc}")
            if glfw is not None:
                if self._window is not None:
                    glfw.destroy_window(self._window)
                    self._window = None
                if self._glfw_acquired:
                    from .glfw_support import release

                    release(glfw)
                    self._glfw_acquired = False
            self.initialized = False


__all__ = ["GPURelightRenderer", "LightingQuality", "QUALITY_PROFILES"]
