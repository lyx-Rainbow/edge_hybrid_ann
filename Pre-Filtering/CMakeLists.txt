cmake_minimum_required(VERSION 3.10)
project(prefiltering LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for batch_query (optional; build still works without it)
find_package(OpenMP)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Source files
set(PREFILTERING_SOURCES
    src/cnpy.cpp
    src/prefiltering_index.cpp
    src/main.cpp
)

# Header files (for IDE integration, not required for build)
set(PREFILTERING_HEADERS
    src/cnpy.h
    src/config.h
    src/distance.h
    src/predicate.h
    src/prefiltering_index.h
)

# Executable
add_executable(prefiltering ${PREFILTERING_SOURCES} ${PREFILTERING_HEADERS})

# Include directories
target_include_directories(prefiltering PRIVATE src)

# Link OpenMP if available
if(OpenMP_FOUND)
    target_link_libraries(prefiltering PRIVATE OpenMP::OpenMP_CXX)
endif()
