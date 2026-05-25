#include "bindings/common.h"

namespace pyrxmesh_py {

void register_module_core(py::module_& m)
{
    m.def(
        "abi_version",
        []() { return plugin_abi_version; },
        "Return the PyRXMesh plugin ABI version.");
    m.def(
        "build_config_tag",
        []() { return std::string(build_config_tag); },
        "Return the RXMesh/PyRXMesh build configuration tag.");

    py::module_ location = add_constants_module(
        m, "Location", "RXMesh memory location bitmask constants.");
    location.attr("NONE")   = py::int_(static_cast<int>(rxmesh::LOCATION_NONE));
    location.attr("HOST")   = py::int_(static_cast<int>(rxmesh::HOST));
    location.attr("DEVICE") = py::int_(static_cast<int>(rxmesh::DEVICE));
    location.attr("ALL")    = py::int_(static_cast<int>(rxmesh::LOCATION_ALL));

    py::module_ layout = add_constants_module(
        m, "Layout", "RXMesh attribute memory layout constants.");
    layout.attr("AoS") = py::int_(static_cast<int>(rxmesh::AoS));
    layout.attr("SoA") = py::int_(static_cast<int>(rxmesh::SoA));

    py::enum_<ElementKind>(m, "ElementKind")
        .value("Vertex", ElementKind::Vertex)
        .value("Edge", ElementKind::Edge)
        .value("Face", ElementKind::Face);

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int8", DType::Int8);

    py::enum_<rxmesh::Op>(m, "Op")
        .value("INVALID", rxmesh::Op::INVALID)
        .value("V", rxmesh::Op::V)
        .value("E", rxmesh::Op::E)
        .value("F", rxmesh::Op::F)
        .value("VV", rxmesh::Op::VV)
        .value("VE", rxmesh::Op::VE)
        .value("VF", rxmesh::Op::VF)
        .value("FV", rxmesh::Op::FV)
        .value("FE", rxmesh::Op::FE)
        .value("FF", rxmesh::Op::FF)
        .value("EV", rxmesh::Op::EV)
        .value("EE", rxmesh::Op::EE)
        .value("EF", rxmesh::Op::EF)
        .value("EVDiamond", rxmesh::Op::EVDiamond);

    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("TRACE", spdlog::level::trace)
        .value("DEBUG", spdlog::level::debug)
        .value("INFO", spdlog::level::info)
        .value("WARN", spdlog::level::warn)
        .value("ERROR", spdlog::level::err)
        .value("CRITICAL", spdlog::level::critical)
        .value("OFF", spdlog::level::off);

    m.def(
        "init",
        [](int device_id, spdlog::level::level_enum log_level) {
            rxmesh::rx_init(device_id, log_level);
        },
        py::arg("device_id") = 0,
        py::arg("log_level") = spdlog::level::info,
        "Initialize RXMesh logging and select a CUDA device.");

    m.def("show", &show_polyscope, "Open the Polyscope viewer.");
}

}  // namespace pyrxmesh_py
