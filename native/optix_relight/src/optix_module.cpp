// Optional native boundary for the RTX renderer. A production OptiX build
// replaces this boundary with BLAS/TLAS and ray-generation hit programs.
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_p123_optix, m) {
    m.def("create_renderer", [](const std::string&) -> py::object {
        throw std::runtime_error(
            "P123 OptiX pipeline is not included in this host build; use raster fallback");
    });
}
