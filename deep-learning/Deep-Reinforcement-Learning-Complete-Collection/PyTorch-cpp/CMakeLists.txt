cmake_minimum_required(VERSION 3.10.0)
project(CppRl
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "Reinforcement learning in C++ using PyTorch"
)

# Project-wide properties
set(CMAKE_CXX_STANDARD 17)

# Cppcheck
list(APPEND CPPCHECK_ARGS
    --enable=warning
    --std=c++14
    --verbose
    --force
    --quiet
    --inline-suppr
    --error-exitcode=1
    --language=c++
    --config-exclude=${CMAKE_CURRENT_LIST_DIR}/src/third_party
    --config-exclude=${CMAKE_CURRENT_LIST_DIR}/lib
    --suppressions-list=${CMAKE_CURRENT_LIST_DIR}/CppCheckSuppressions.txt 
    -I ${CMAKE_CURRENT_LIST_DIR}/src 
    -I ${CMAKE_CURRENT_LIST_DIR}/include 
    -I ${CMAKE_CURRENT_LIST_DIR}/lib/spdlog/include 
    -I ${CMAKE_CURRENT_LIST_DIR}/example 
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${CMAKE_CURRENT_LIST_DIR}/example
)

add_custom_target(
    check
    COMMAND cppcheck ${CPPCHECK_ARGS}
    COMMENT "Running Cppcheck"
)

# Dependencies
## PyTorch
find_package(Torch REQUIRED)
if (TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
endif()
## Spdlog
add_subdirectory(lib/spdlog)

# Define targets
add_library(cpprl STATIC "")
target_compile_definitions(cpprl PRIVATE DOCTEST_CONFIG_DISABLE)
option(CPPRL_BUILD_TESTS "Whether or not to build the CppRl tests" ON)
if (CPPRL_BUILD_TESTS)
    add_executable(cpprl_tests "")
endif(CPPRL_BUILD_TESTS)

# Enable all warnings
if(MSVC)
    target_compile_options(cpprl PRIVATE /W4 /WX)
else(MSVC)
    target_compile_options(cpprl PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

# Includes
set(CPPRL_INCLUDE_DIRS
    include
    src
    lib/spdlog/include
    ${TORCH_INCLUDE_DIRS}
)
target_include_directories(cpprl PRIVATE ${CPPRL_INCLUDE_DIRS})
target_include_directories(cpprl_tests PRIVATE ${CPPRL_INCLUDE_DIRS})

# Linking
target_link_libraries(cpprl torch ${TORCH_LIBRARIES})
target_link_libraries(cpprl_tests torch ${TORCH_LIBRARIES})

# Example
add_subdirectory(example)

# Recurse into source tree
add_subdirectory(src)