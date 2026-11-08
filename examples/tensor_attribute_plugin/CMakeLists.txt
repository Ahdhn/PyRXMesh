cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(rxmesh_tensor_attribute LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pyrxmesh.cmake_dir
    OUTPUT_VARIABLE pyrxmesh_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(pyrxmesh CONFIG REQUIRED PATHS "${pyrxmesh_DIR}" NO_DEFAULT_PATH)
pyrxmesh_add_plugin(_rxmesh_tensor_attribute src/rxmesh_tensor_attribute.cu)
install(TARGETS _rxmesh_tensor_attribute
    LIBRARY DESTINATION rxmesh_tensor_attribute COMPONENT python
    RUNTIME DESTINATION rxmesh_tensor_attribute COMPONENT python
)
