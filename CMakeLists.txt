cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYRXMESH_PYPROJECT_TOML)
string(REGEX MATCH
    "\\[project\\][^[]*version[ \t]*=[ \t]*\"([^\"]+)\""
    PYRXMESH_VERSION_MATCH
    "${PYRXMESH_PYPROJECT_TOML}"
)
set(PYRXMESH_VERSION "${CMAKE_MATCH_1}")

project(PyRXMesh LANGUAGES C CXX CUDA)
# Increment when the compiled custom-plugin capsule/API contract changes incompatibly and old plugins will no longer work with the new PyRXMesh version. This is separate from the overall project version, which can be incremented for any change including compatible ones.
set(PYRXMESH_PLUGIN_ABI_VERSION 2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

set(PYRXMESH_RXMESH_GIT_REPO "https://github.com/owensgroup/RXMesh.git" CACHE STRING "RXMesh Git repository URL")
set(PYRXMESH_RXMESH_GIT_TAG "main" CACHE STRING "RXMesh Git tag, branch, or commit to build against")
set(PYRXMESH_RXMESH_SOURCE_DIR "" CACHE PATH "Optional local RXMesh source checkout. If set, Git fetch is skipped.")
set(PYRXMESH_USE_POLYSCOPE "ON" CACHE BOOL "Build RXMesh with Polyscope visualization support")
set(PYRXMESH_BUILD_RXMESH_TESTS "OFF" CACHE BOOL "Build RXMesh tests when building PyRXMesh")
set(PYRXMESH_BUILD_RXMESH_APPS "OFF" CACHE BOOL "Build RXMesh apps when building PyRXMesh")
set(PYRXMESH_USE_DOUBLE "OFF" CACHE BOOL "Build RXMesh with double precision coordinate input")
set(PYRXMESH_USE_CUDSS "OFF" CACHE BOOL "Build RXMesh with cuDSS support")
set(PYRXMESH_USE_SUITESPARSE "OFF" CACHE BOOL "Build RXMesh with SuiteSparse support")
set(PYRXMESH_INSTALL_PLUGIN_SDK "ON" CACHE BOOL "Install headers, libraries, and CMake files needed to build PyRXMesh plugins")


string(REPLACE "\\" "/" PYRXMESH_RXMESH_SOURCE_DIR_TAG "${PYRXMESH_RXMESH_SOURCE_DIR}")
set(PYRXMESH_CUDA_ARCHITECTURES_TAG "${CMAKE_CUDA_ARCHITECTURES}")
string(REPLACE ";" "," PYRXMESH_CUDA_ARCHITECTURES_TAG "${PYRXMESH_CUDA_ARCHITECTURES_TAG}")

# Captures build options that plugins must match to safely link against this PyRXMesh/RXMesh install. Used by plugins to check compatibility at runtime and refuse to load if incompatible, preventing hard-to-debug crashes due to ABI incompatibility.
set(PYRXMESH_BUILD_CONFIG_TAG
    "abi=${PYRXMESH_PLUGIN_ABI_VERSION},pyrxmesh=${PYRXMESH_VERSION},rxmesh_repo=${PYRXMESH_RXMESH_GIT_REPO},rxmesh_tag=${PYRXMESH_RXMESH_GIT_TAG},rxmesh_source=${PYRXMESH_RXMESH_SOURCE_DIR_TAG},polyscope=${PYRXMESH_USE_POLYSCOPE},double=${PYRXMESH_USE_DOUBLE},cudss=${PYRXMESH_USE_CUDSS},suitesparse=${PYRXMESH_USE_SUITESPARSE},cuda_arch=${PYRXMESH_CUDA_ARCHITECTURES_TAG}"
)

# RXMesh uses these option names internally. Force them before adding RXMesh so PyRXMesh controls the dependency build in both Git and local-source modes.
set(RX_BUILD_TESTS ${PYRXMESH_BUILD_RXMESH_TESTS} CACHE BOOL "" FORCE)
set(RX_BUILD_APPS ${PYRXMESH_BUILD_RXMESH_APPS} CACHE BOOL "" FORCE)
set(RX_USE_POLYSCOPE ${PYRXMESH_USE_POLYSCOPE} CACHE BOOL "" FORCE)
set(RX_USE_DOUBLE ${PYRXMESH_USE_DOUBLE} CACHE BOOL "" FORCE)
set(RX_USE_CUDSS ${PYRXMESH_USE_CUDSS} CACHE BOOL "" FORCE)
set(RX_USE_SUITESPARSE ${PYRXMESH_USE_SUITESPARSE} CACHE BOOL "" FORCE)
set(RXMESH_INSTALL_INCLUDEDIR "pyrxmesh/include" CACHE STRING "" FORCE)
set(RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR "pyrxmesh/third_party/include" CACHE STRING "" FORCE)
set(RXMESH_INSTALL_LIBDIR "pyrxmesh/lib" CACHE STRING "" FORCE)
set(RXMESH_INSTALL_BINDIR "pyrxmesh/bin" CACHE STRING "" FORCE)
set(RXMESH_INSTALL_CMAKEDIR "pyrxmesh/cmake/RXMesh" CACHE STRING "" FORCE)
if(PYRXMESH_INSTALL_PLUGIN_SDK)
    set(RXMESH_INSTALL_COMPONENT "python" CACHE STRING "" FORCE)
else()
    set(RXMESH_INSTALL_COMPONENT "plugin_sdk" CACHE STRING "" FORCE)
endif()

if(PYRXMESH_RXMESH_SOURCE_DIR)
    get_filename_component(PYRXMESH_RXMESH_SOURCE_DIR_ABS "${PYRXMESH_RXMESH_SOURCE_DIR}" ABSOLUTE)
    if(NOT EXISTS "${PYRXMESH_RXMESH_SOURCE_DIR_ABS}/CMakeLists.txt")
        message(FATAL_ERROR "PYRXMESH_RXMESH_SOURCE_DIR does not point to an RXMesh source tree: ${PYRXMESH_RXMESH_SOURCE_DIR_ABS}")
    endif()
    add_subdirectory("${PYRXMESH_RXMESH_SOURCE_DIR_ABS}" "${CMAKE_BINARY_DIR}/_deps/rxmesh-build")
else()
    FetchContent_Declare(rxmesh
        GIT_REPOSITORY "${PYRXMESH_RXMESH_GIT_REPO}"
        GIT_TAG "${PYRXMESH_RXMESH_GIT_TAG}"
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(rxmesh)
endif()

set_target_properties(RXMesh PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_rxmesh MODULE
    src/bindings.cu
    src/bindings/core.cu
    src/bindings/geometry.cu
    src/bindings/handles.cu
    src/bindings/mesh.cu
    src/bindings/common.h
    src/bindings/dlpack_minimal.h

    src/bindings/py_attributes.cu
    src/bindings/py_attribute.h

    src/bindings/py_dense_matrix.cu
    src/bindings/py_dense_matrix.h
    src/bindings/dense_matrix_dlpack.h

    src/bindings/py_sparse_matrix.cu
    src/bindings/py_sparse_matrix.h
    src/bindings/py_solvers.cu
    src/bindings/py_solvers.h
    src/bindings/py_hessian_sparse_matrix.h
    src/bindings/py_jacobian_sparse_matrix.h
    src/bindings/sparse_matrix_csr.h
    src/bindings/sparse_matrix_dlpack.h
)

target_link_libraries(_rxmesh PRIVATE RXMesh)
target_include_directories(_rxmesh
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(_rxmesh PRIVATE cxx_std_17)
target_compile_definitions(_rxmesh
    PRIVATE PYRXMESH_PLUGIN_ABI_VERSION=${PYRXMESH_PLUGIN_ABI_VERSION}
    PRIVATE PYRXMESH_BUILD_CONFIG="${PYRXMESH_BUILD_CONFIG_TAG}"
)
set_target_properties(_rxmesh PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(UNIX AND NOT APPLE)
    set_target_properties(_rxmesh PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;$ORIGIN/bin"
    )
endif()

if(WIN32)
    target_compile_definitions(_rxmesh
        PRIVATE _USE_MATH_DEFINES
        PRIVATE NOMINMAX
        PRIVATE _CRT_SECURE_NO_WARNINGS
    )
endif()

if(PYRXMESH_USE_CUDSS)
    target_compile_definitions(_rxmesh PRIVATE USE_CUDSS)
endif()

install(TARGETS _rxmesh
    LIBRARY DESTINATION pyrxmesh COMPONENT python
    RUNTIME DESTINATION pyrxmesh COMPONENT python
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pyrxmeshConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pyrxmeshConfig.cmake"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/build_config.json"
    @ONLY
)

if(PYRXMESH_INSTALL_PLUGIN_SDK)
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/pyrxmesh/plugin_api.h"
        DESTINATION pyrxmesh/include/pyrxmesh
        COMPONENT python
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pyrxmeshConfig.cmake"
        DESTINATION pyrxmesh/cmake
        COMPONENT python
    )
endif()
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/build_config.json"
    DESTINATION pyrxmesh
    COMPONENT python
)
