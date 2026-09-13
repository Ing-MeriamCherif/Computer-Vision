"""Low-resolution NVIDIA OptiX visibility rays for RTX hardware."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import time

import numpy as np


def _include_dir() -> Path:
    candidates = [
        os.environ.get("OPTIX_INCLUDE_DIR"),
        r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include",
        r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.1.0\include",
        str(Path(os.environ.get("TEMP", "")) / "nrw-otk-pyoptix-v130-msvc" / "build-msvc-5" / "_deps" / "optix_headers-src" / "include"),
    ]
    for item in candidates:
        if item and (Path(item) / "optix.h").is_file():
            return Path(item)
    raise RuntimeError("OptiX headers not found; set OPTIX_INCLUDE_DIR to the NVIDIA OptiX SDK include folder")


def _nvrtc_check(result, program=None):
    if result[0].value:
        if program is not None:
            _, size = __import__("cuda.bindings.nvrtc", fromlist=["nvrtc"]).nvrtcGetProgramLogSize(program)
            log = b" " * size
            __import__("cuda.bindings.nvrtc", fromlist=["nvrtc"]).nvrtcGetProgramLog(program, log)
            raise RuntimeError(log.decode("utf-8", "replace"))
        raise RuntimeError(f"NVRTC failed with code {result[0].value}")
    return result[1] if len(result) == 2 else None


def _compile_ptx(nvrtc, source: Path, include: Path) -> bytes:
    cuda_root = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"))
    options = [b"-use_fast_math", b"-std=c++11", b"-rdc=true", f"-I{include}".encode(), f"-I{cuda_root / 'include'}".encode()]
    program = _nvrtc_check(nvrtc.nvrtcCreateProgram(source.read_bytes(), str(source).encode(), 0, [], []))
    _nvrtc_check(nvrtc.nvrtcCompileProgram(program, len(options), options), program)
    size = _nvrtc_check(nvrtc.nvrtcGetPTXSize(program))
    ptx = b" " * size
    _nvrtc_check(nvrtc.nvrtcGetPTX(program, ptx))
    return ptx


class OptixShadowRenderer:
    """Trace one hard visibility ray per coarse depth sample and light."""

    def __init__(self, width: int = 96, height: int = 54):
        import cupy as cp
        import optix
        from cuda.bindings import nvrtc

        cp.cuda.runtime.free(0)
        self.cp, self.optix = cp, optix
        self.width, self.height = max(32, int(width)), max(18, int(height))
        self.faces = self._make_indices(self.width, self.height)
        source = Path(__file__).with_name("optix_shadow.cu")
        ptx = _compile_ptx(nvrtc, source, _include_dir())
        self.ctx = optix.deviceContextCreate(0, optix.DeviceContextOptions(logCallbackLevel=2))
        self._create_pipeline(ptx)
        self.vertices = cp.zeros(((self.width - 1) * (self.height - 1) * 6, 3), dtype=cp.float32)
        self.points = cp.zeros((self.width * self.height, 3), dtype=cp.float32)
        self.normals = cp.zeros_like(self.points)
        indices = np.arange(self.vertices.shape[0], dtype=np.uint32).reshape(-1, 3)
        self.indices = cp.asarray(indices, dtype=cp.uint32)
        self.visibility = cp.ones((2, self.height, self.width), dtype=cp.float32)
        self.param_dtype = self._param_dtype()
        self.params = cp.empty(self.param_dtype.itemsize, dtype=cp.uint8)
        self.gas_handle, self.gas_buffer, self.gas_temp = self._build_gas()
        self.sbt = self._create_sbt()

    @staticmethod
    def _param_dtype():
        return np.dtype([
            ("traversable", np.uint64), ("points", np.uint64), ("normals", np.uint64),
            ("visibility", np.uint64), ("width", np.uint32), ("height", np.uint32),
            ("light_count", np.uint32), ("light_positions", np.float32, (6,)),
            ("source_radii", np.float32, (2,)), ("self_eps", np.float32, (2,)),
            ("emitter_exclusion", np.float32, (2,)),
        ], align=True)

    def _create_pipeline(self, ptx):
        o = self.optix
        po = o.PipelineCompileOptions(usesMotionBlur=False,
            traversableGraphFlags=int(o.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=1, numAttributeValues=2, exceptionFlags=int(o.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params", usesPrimitiveTypeFlags=o.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
        mo = o.ModuleCompileOptions(maxRegisterCount=o.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=o.COMPILE_OPTIMIZATION_DEFAULT, debugLevel=o.COMPILE_DEBUG_LEVEL_NONE)
        module, _ = self.ctx.moduleCreate(mo, po, ptx)
        rg = o.ProgramGroupDesc(); rg.raygenModule = module; rg.raygenEntryFunctionName = "__raygen__shadow"
        ms = o.ProgramGroupDesc(); ms.missModule = module; ms.missEntryFunctionName = "__miss__shadow"
        hg = o.ProgramGroupDesc(); hg.hitgroupModuleCH = module; hg.hitgroupEntryFunctionNameCH = "__closesthit__shadow"
        groups = [self.ctx.programGroupCreate([desc])[0][0] for desc in (rg, ms, hg)]
        link = o.PipelineLinkOptions(); link.maxTraceDepth = 1
        self.pipeline = self.ctx.pipelineCreate(po, link, groups, "")
        stack = o.StackSizes()
        for group in groups:
            o.util.accumulateStackSizes(group, stack, self.pipeline)
        direct, direct_from_traversal, continuation = o.util.computeStackSizes(stack, 1, 0, 0)
        self.pipeline.setStackSize(direct, direct_from_traversal, continuation, 1)
        self.groups = groups

    @staticmethod
    def _make_indices(width, height=None):
        if height is None:
            height = width.height
            width = width.width
        indices = np.empty(((width - 1) * (height - 1) * 2, 3), dtype=np.uint32)
        cursor = 0
        for y in range(height - 1):
            for x in range(width - 1):
                a = y * width + x
                b, c, d = a + 1, a + width, a + width + 1
                indices[cursor] = (a, c, b); indices[cursor + 1] = (b, c, d)
                cursor += 2
        return indices

    def _triangle_input(self):
        o = self.optix
        tri = o.BuildInputTriangleArray()
        tri.vertexFormat = o.VERTEX_FORMAT_FLOAT3
        tri.numVertices = int(self.vertices.shape[0])
        tri.vertexBuffers = [self.vertices.data.ptr]
        tri.indexFormat = o.INDICES_FORMAT_UNSIGNED_INT3
        tri.numIndexTriplets = int(self.indices.shape[0])
        tri.indexBuffer = self.indices.data.ptr
        tri.flags = [o.GEOMETRY_FLAG_NONE]
        tri.numSbtRecords = 1
        return tri

    def _build_gas(self):
        o = self.optix
        options = o.AccelBuildOptions(buildFlags=int(o.BUILD_FLAG_ALLOW_UPDATE | o.BUILD_FLAG_PREFER_FAST_TRACE),
                                      operation=o.BUILD_OPERATION_BUILD)
        sizes = self.ctx.accelComputeMemoryUsage([options], [self._triangle_input()])
        temp_bytes = max(int(sizes.tempSizeInBytes), int(getattr(sizes, "tempUpdateSizeInBytes", 0)))
        temp, output = self.cp.cuda.alloc(temp_bytes), self.cp.cuda.alloc(sizes.outputSizeInBytes)
        handle = self.ctx.accelBuild(0, [options], [self._triangle_input()], temp.ptr, temp_bytes,
                                    output.ptr, sizes.outputSizeInBytes, [])
        self.gas_options, self.gas_sizes = options, sizes
        return handle, output, temp

    def _update_gas(self):
        o = self.optix
        options = o.AccelBuildOptions(buildFlags=int(o.BUILD_FLAG_ALLOW_UPDATE | o.BUILD_FLAG_PREFER_FAST_TRACE),
                                      operation=o.BUILD_OPERATION_UPDATE)
        handle = self.ctx.accelBuild(0, [options], [self._triangle_input()], self.gas_temp.ptr,
            max(int(self.gas_sizes.tempSizeInBytes), int(self.gas_sizes.tempUpdateSizeInBytes)), self.gas_buffer.ptr, self.gas_sizes.outputSizeInBytes, [])
        self.gas_options = options
        return handle

    def _create_sbt(self):
        o, cp = self.optix, self.cp
        dtype = np.dtype([("header", "u1", (o.SBT_RECORD_HEADER_SIZE,))], align=True)
        records = []
        for group in self.groups:
            record = np.zeros(1, dtype=dtype)
            o.sbtRecordPackHeader(group, record)
            records.append(cp.asarray(record))
        return o.ShaderBindingTable(raygenRecord=records[0].data.ptr,
            missRecordBase=records[1].data.ptr, missRecordStrideInBytes=dtype.itemsize, missRecordCount=1,
            hitgroupRecordBase=records[2].data.ptr, hitgroupRecordStrideInBytes=dtype.itemsize, hitgroupRecordCount=1), records

    def _mesh(self, depth, valid, normals, camera):
        h, w = depth.shape
        xs = np.rint(np.linspace(0, w - 1, self.width)).astype(int)
        ys = np.rint(np.linspace(0, h - 1, self.height)).astype(int)
        z = np.asarray(depth[np.ix_(ys, xs)], dtype=np.float32)
        mask = np.asarray(valid[np.ix_(ys, xs)], dtype=bool) & np.isfinite(z) & (z > 1e-4)
        normal = np.asarray(normals[np.ix_(ys, xs)], dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack(((xx - camera.cx) * z / camera.fx, (yy - camera.cy) * z / camera.fy, z), axis=-1)
        points[~mask] = 0
        normal[~mask] = (0, 0, 1)
        faces = self.faces
        face_xyz = points.reshape(-1, 3)[faces]
        face_valid = mask.reshape(-1)[faces].all(axis=1)
        span = face_xyz[:, :, 2].max(axis=1) - face_xyz[:, :, 2].min(axis=1)
        face_valid &= span < np.maximum(0.12, face_xyz[:, :, 2].mean(axis=1) * 0.08)
        gas_vertices = face_xyz.copy()
        gas_vertices[~face_valid] = gas_vertices[~face_valid, :1]
        return (np.ascontiguousarray(gas_vertices.reshape(-1, 3), dtype=np.float32),
                np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32),
                np.ascontiguousarray(normal.reshape(-1, 3), dtype=np.float32))

    def render(self, depth, valid, normals, camera, lights):
        started = time.perf_counter()
        vertices, points, normals = self._mesh(depth, valid, normals, camera)
        cp = self.cp
        self.vertices.set(vertices); self.points.set(points); self.normals.set(normals)
        handle = self._update_gas()
        positions = np.zeros((2, 3), dtype=np.float32)
        radii = np.full(2, 0.025, dtype=np.float32)
        epsilons = np.full(2, 0.002, dtype=np.float32)
        exclusions = np.zeros(2, dtype=np.float32)
        count = min(len(lights), 2)
        for i, light in enumerate(lights[:count]):
            positions[i] = light.position_camera
            radii[i] = max(float(getattr(light, "source_radius_m", 0.025)), 0.0)
            epsilons[i] = max(float(getattr(light, "self_intersection_epsilon_m", 0.002)), 1e-4)
            if bool(getattr(light, "is_palm_attached", False)):
                exclusions[i] = 0.06
        host_params = np.zeros(1, dtype=self.param_dtype)
        host_params[0] = (handle, self.points.data.ptr, self.normals.data.ptr,
                          self.visibility.data.ptr, self.width, self.height, count, positions.reshape(-1), radii, epsilons, exclusions)
        self.params.set(host_params.view(np.uint8))
        optix = self.optix
        optix.launch(self.pipeline, 0, self.params.data.ptr, self.param_dtype.itemsize,
                     self.sbt[0], self.width, self.height, 2)
        cp.cuda.runtime.deviceSynchronize()
        output = cp.asnumpy(self.visibility).transpose(1, 2, 0)
        return output, (time.perf_counter() - started) * 1000.0


def create_optix_renderer(width: int = 96, height: int = 54):
    try:
        return OptixShadowRenderer(width, height), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
