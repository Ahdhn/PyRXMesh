cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(rxmesh_edge_lengths LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pyrxmesh.cmake_dir
    OUTPUT_VARIABLE pyrxmesh_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

find_package(pyrxmesh CONFIG REQUIRED PATHS "${pyrxmesh_DIR}" NO_DEFAULT_PATH)

pyrxmesh_add_plugin(_rxmesh_edge_lengths src/rxmesh_edge_lengths.cu)

install(TARGETS _rxmesh_edge_lengths
    LIBRARY DESTINATION rxmesh_edge_lengths COMPONENT python
    RUNTIME DESTINATION rxmesh_edge_lengths COMPONENT python
)
