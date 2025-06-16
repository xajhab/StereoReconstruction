cmake_minimum_required(VERSION 3.10)
project(RootBA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Check OpenCV version
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(WARNING "OpenCV ${OpenCV_VERSION} found. Version 4.0+ recommended.")
endif()

# List all cpp files explicitly
set(SOURCES
    src/main.cpp
    src/disparity.cpp
    src/depth.cpp
    src/denseMatching.cpp
    src/8point.cpp
    src/meshReconstruction.cpp
)

# Check all files exist
foreach(source_file ${SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        message(FATAL_ERROR "Required source file not found: ${source_file}")
    endif()
endforeach()

# Find all header files
file(GLOB_RECURSE HEADERS 
    "src/*.h"
    "*.h"
)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# Create executable file
add_executable(main ${SOURCES})

# Link libraries
target_link_libraries(main ${OpenCV_LIBS})

# Compile options
target_compile_options(main PRIVATE 
    -Wall -Wextra -O2
    $<$<CONFIG:Debug>:-g>
)

# Output debug information
message(STATUS "Found sources: ${SOURCES}")
message(STATUS "Found headers: ${HEADERS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")