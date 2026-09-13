"""OpenGL 3.3 screen-space relighting for P123 GeometryState frames."""

from __future__ import annotations

from dataclasses import dataclass
import time
import threading
from typing import Any

import numpy as np

from .lighting import LightState
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
uniform vec2 uResolution;
uniform vec4 uCamera; // fx, fy, cx, cy
uniform int uLightCount;
uniform vec3 uLightPosition[2];
uniform vec3 uLightColor[2];
uniform vec4 uLightPower[2]; // intensity, confidence, range, source radius
uniform int uShadowRays;
uniform int uShadowSteps;
uniform int uHistoryAllowed;
uniform float uHistoryWeight;
uniform float uAmbient;

vec3 toLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

float rayVisibility(vec3 p, vec3 n, vec3 lightPos, float radius) {
    const vec2 offsets[6] = vec2[6](
        vec2(-0.65,-0.65), vec2(0.65,-0.65), vec2(-0.65,0.65),
        vec2(0.65,0.65), vec2(0.0,-0.75), vec2(0.0,0.75));
    float visibility = 0.0;
    for (int rayIndex=0; rayIndex<6; ++rayIndex) {
        if (rayIndex >= uShadowRays) break;
        vec3 emitter = lightPos + vec3(offsets[rayIndex] * radius, 0.0);
        vec3 start = p + n * max(0.002, 0.004 * p.z);
        float rayVisible = 1.0;
        for (int stepIndex=0; stepIndex<10; ++stepIndex) {
            if (stepIndex >= uShadowSteps) break;
            float t = mix(0.07, 0.94, (float(stepIndex) + 0.5) / float(max(uShadowSteps, 1)));
            vec3 s = mix(start, emitter, t);
            if (s.z <= 1e-4) continue;
            vec2 pixel = vec2(uCamera.x * s.x / s.z + uCamera.z,
                              uCamera.y * s.y / s.z + uCamera.w);
            vec2 uv = pixel / uResolution;
            if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
            float sceneZ = texture(uDepth, uv).r;
            float bias = max(0.005, 0.012 * s.z);
            if (sceneZ > 1e-5 && sceneZ < s.z - bias) {
                rayVisible = 0.20;
                break;
            }
        }
        visibility += rayVisible;
    }
    return visibility / float(max(uShadowRays, 1));
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
    vec3 n = normalize(texture(uNormal, vUv).xyz);
    float conf = clamp(texture(uConfidence, vUv).r, 0.0, 1.0);
    float normalConfidence = smoothstep(0.05, 0.45, length(texture(uNormal, vUv).xyz));
    vec3 direct = vec3(0.0);
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
        float specular = pow(max(dot(n, h), 0.0), 36.0);
        float rangeM = max(uLightPower[i].z, 0.01);
        float attenuation = uLightPower[i].x / (1.0 + pow(distanceToLight / rangeM, 2.0));
        float visibility = rayVisibility(p, n, uLightPosition[i], uLightPower[i].w);
        if (uHistoryAllowed != 0 && uHistoryWeight > 0.0) {
            float previousZ = texture(uPrevDepth, vUv).r;
            float previousVisibility = texture(uPrevShadow, vUv).r;
            if (previousZ > 1e-5 && abs(previousZ - z) < max(0.018, 0.025 * z)) {
                visibility = mix(visibility, previousVisibility, uHistoryWeight);
            }
        }
        visibilitySum += visibility;
        direct += uLightColor[i] * (diffuse * 0.92 + specular * 0.12) * attenuation * visibility * uLightPower[i].y * conf * normalConfidence;
    }
    if (uLightCount > 0) visibilityMean = visibilitySum / float(uLightCount);
    vec3 irradiance = vec3(max(uAmbient, 0.05)) + direct;
    vec3 shaded = base * (irradiance / max(uAmbient, 0.05));
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
uniform int uVolumeSteps;
uniform int uVolShadowSteps;
uniform int uHistoryAllowed;
uniform float uHistoryWeight;
uniform float uDensity;

float lightVisibility(vec3 samplePos, vec3 lightPos) {
    float visible = 1.0;
    vec3 start = samplePos + (lightPos - samplePos) * 0.05;
    for (int i=0; i<6; ++i) {
        if (i >= uVolShadowSteps) break;
        float t = mix(0.12, 0.92, (float(i) + 0.5) / float(max(uVolShadowSteps, 1)));
        vec3 s = mix(start, lightPos, t);
        if (s.z <= 1e-4) continue;
        vec2 pixel = vec2(uCamera.x * s.x / s.z + uCamera.z,
                          uCamera.y * s.y / s.z + uCamera.w);
        vec2 uv = pixel / uResolution;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
        float sceneZ = texture(uDepth, uv).r;
        if (sceneZ > 1e-5 && sceneZ < s.z - max(0.008, 0.014 * s.z)) {
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
    for (int sampleIndex=0; sampleIndex<12; ++sampleIndex) {
        if (sampleIndex >= uVolumeSteps) break;
        float f = (float(sampleIndex) + 0.5) / float(max(uVolumeSteps, 1));
        float sampleZ = mix(0.15, max(maxZ, 0.2), f);
        vec3 s = ray * sampleZ;
        if (valid > 0.5 && surfaceZ > 1e-5 && sampleZ > surfaceZ - max(0.004, 0.008 * surfaceZ)) continue;
        for (int lightIndex=0; lightIndex<2; ++lightIndex) {
            if (lightIndex >= uLightCount) break;
            vec3 delta = uLightPosition[lightIndex] - s;
            float distanceSq = max(dot(delta, delta), 0.0025);
            float rangeM = max(uLightPower[lightIndex].z, 0.01);
            float rangeWeight = exp(-0.5 * distanceSq / (rangeM * rangeM));
            float visibility = lightVisibility(s, uLightPosition[lightIndex]);
            float scatter = uDensity * uLightPower[lightIndex].x * uLightPower[lightIndex].y;
            haze += uLightColor[lightIndex] * rangeWeight * visibility * scatter;
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
        vec2 center = vec2(uCamera.x * p.x / p.z + uCamera.z,
                           uCamera.y * p.y / p.z + uCamera.w);
        float radius = clamp(uCamera.x * max(uLightPower[i].w, 0.004) / p.z, 3.5, 14.0);
        float d = length(pixel - center);
        float outer = exp(-0.5 * pow(d / max(radius * 1.35, 1.0), 2.0));
        float inner = exp(-0.5 * pow(d / max(radius * 0.58, 1.0), 2.0));
        float core = exp(-0.5 * pow(d / max(radius * 0.23, 1.0), 2.0));
        float power = clamp(uLightPower[i].x * uLightPower[i].y, 0.0, 2.0);
        float sceneZ = texture(uDepth, vUv).r;
        float haloVisibility = sceneZ > 1e-5 && sceneZ + max(0.025, 0.04 * p.z) < p.z ? 0.28 : 1.0;
        color += uLightColor[i] * max(outer-inner, 0.0) * 0.035 * power * haloVisibility;
        color += uLightColor[i] * inner * 0.075 * power * mix(0.55, 1.0, haloVisibility);
        color = mix(color, mix(uLightColor[i], vec3(1.0,0.97,0.90),0.68), core * 0.34);
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
        self.initialized = False
        self.gl_version = "unknown"
        self.last_stats: dict[str, float | str] = {}
        self.set_quality(quality)
        self._initialize()

    def set_quality(self, quality: str) -> None:
        value = str(quality).lower()
        if value not in QUALITY_PROFILES:
            raise ValueError(f"unknown lighting quality '{quality}'")
        self.quality_name = value
        self.quality = QUALITY_PROFILES[value]
        if getattr(self, "initialized", False):
            self._history_key = None

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
            self._programs = {
                "surface": self._link_program(_VERTEX_SHADER, _SURFACE_SHADER),
                "volume": self._link_program(_VERTEX_SHADER, _VOLUME_SHADER),
                "composite": self._link_program(_VERTEX_SHADER, _COMPOSITE_SHADER),
            }
            self._vao = int(gl.glGenVertexArrays(1))
            gl.glBindVertexArray(self._vao)
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
        shaders = []
        for kind, source in ((gl.GL_VERTEX_SHADER, vertex_source), (gl.GL_FRAGMENT_SHADER, fragment_source)):
            shader = gl.glCreateShader(kind)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                message = gl.glGetShaderInfoLog(shader).decode("utf-8", "replace")
                gl.glDeleteShader(shader)
                raise RuntimeError(f"GLSL compile failed: {message}")
            shaders.append(shader)
        program = gl.glCreateProgram()
        for shader in shaders:
            gl.glAttachShader(program, shader)
        gl.glLinkProgram(program)
        for shader in shaders:
            gl.glDeleteShader(shader)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            message = gl.glGetProgramInfoLog(program).decode("utf-8", "replace")
            gl.glDeleteProgram(program)
            raise RuntimeError(f"GLSL link failed: {message}")
        return int(program)

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
        for i in range(2):
            self._texture(f"shadow{i}", width, height, gl.GL_R16F, gl.GL_RED, gl.GL_FLOAT, filtering=linear)
            self._texture(f"history_depth{i}", width, height, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, filtering=nearest)
            self._texture(f"volume{i}", volume_width, volume_height, gl.GL_RGBA16F, gl.GL_RGBA, gl.GL_FLOAT, filtering=linear)
        self._framebuffers["surface"] = int(gl.glGenFramebuffers(1))
        self._framebuffers["volume"] = int(gl.glGenFramebuffers(1))
        self._framebuffers["output"] = int(gl.glGenFramebuffers(1))
        self._history_index = 0
        self._history_key = None

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
        for i, light in enumerate(lights[:2]):
            positions[i] = np.asarray(light.position_camera, dtype=np.float32)
            colors[i] = np.clip(np.asarray(light.color_rgb, dtype=np.float32), 0.0, 1.0)
            powers[i] = (
                max(float(light.intensity), 0.0),
                float(np.clip(light.confidence, 0.0, 1.0)),
                max(float(getattr(light, "range_m", 0.45)), 0.01),
                max(float(getattr(light, "source_radius_m", 0.025)), 0.001),
            )
        gl.glUniform1i(gl.glGetUniformLocation(program, "uLightCount"), min(len(lights), 2))
        gl.glUniform3fv(gl.glGetUniformLocation(program, "uLightPosition[0]"), 2, positions)
        gl.glUniform3fv(gl.glGetUniformLocation(program, "uLightColor[0]"), 2, colors)
        gl.glUniform4fv(gl.glGetUniformLocation(program, "uLightPower[0]"), 2, powers)

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

    def render(
        self,
        rgb: np.ndarray,
        geometry: GeometryState,
        lights: list[LightState] | tuple[LightState, ...],
        *,
        ambient: float = 0.40,
        readback: bool = True,
    ) -> tuple[np.ndarray | None, dict[str, float | str]]:
        with self._render_lock:
            try:
                return self._render_impl(rgb, geometry, lights, ambient=ambient, readback=readback)
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
        active = [light for light in lights if light.enabled and light.confidence > 0 and np.isfinite(light.position_camera).all() and light.position_camera[2] > 0][:2]
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
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uHistoryWeight"), self.quality.history_weight)
        gl.glUniform1f(gl.glGetUniformLocation(surface_program, "uAmbient"), float(ambient))
        self._bind_texture(0, self._textures["rgb"], surface_program, "uRgb")
        self._bind_texture(1, self._textures["depth"], surface_program, "uDepth")
        self._bind_texture(2, self._textures["normal"], surface_program, "uNormal")
        self._bind_texture(3, self._textures["confidence"], surface_program, "uConfidence")
        self._bind_texture(4, self._textures["valid"], surface_program, "uValid")
        self._bind_texture(5, self._textures[f"shadow{previous_index}"], surface_program, "uPrevShadow")
        self._bind_texture(6, self._textures[f"history_depth{previous_index}"], surface_program, "uPrevDepth")
        self._draw(surface_program, width, height)

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
        gl.glUniform1f(gl.glGetUniformLocation(volume_program, "uDensity"), 0.035)
        self._bind_texture(1, self._textures["depth"], volume_program, "uDepth")
        self._bind_texture(2, self._textures["valid"], volume_program, "uValid")
        self._bind_texture(5, self._textures[f"volume{previous_index}"], volume_program, "uPrevVolume")
        self._bind_texture(6, self._textures[f"history_depth{previous_index}"], volume_program, "uPrevDepth")
        self._draw(volume_program, self._volume_width, self._volume_height)

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
        self._draw(composite_program, width, height)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glFinish()
        result = None
        if readback:
            output = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            result = np.frombuffer(output, dtype=np.uint8).reshape(height, width, 3).copy()
        self._history_index = write_index
        total_ms = (time.perf_counter() - render_started) * 1000.0
        self.last_stats = {
            "renderer": "GPU",
            "gpu_render_ms": float(total_ms),
            "lights": float(len(active)),
            "quality": self.quality_name,
            "shadow_quality": f"{self.quality.shadow_rays}x{self.quality.shadow_steps}",
            "volumetric_quality": f"{self.quality.volume_steps}/{self.quality.volume_shadow_steps}",
            "gl_version": self.gl_version,
            "volumetric_resolution": f"{self._volume_width}x{self._volume_height}",
        }
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
