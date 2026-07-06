cmake_minimum_required(VERSION 3.10)
project(diskivf LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for batch_query + K-means parallelism
find_package(OpenMP)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Source files
set(DISKIVF_SOURCES
    src/cnpy.cpp
    src/kmeans.cpp
    src/diskivf_index.cpp
    src/main.cpp
)

# Header files (for IDE integration, not required for build)
set(DISKIVF_HEADERS
    src/cnpy.h
    src/config.h
    src/distance.h
    src/diskivf_index.h
    src/kmeans.h
    src/predicate.h
)

# Executable
add_executable(diskivf ${DISKIVF_SOURCES} ${DISKIVF_HEADERS})

# Include directories
target_include_directories(diskivf PRIVATE src)

# Link OpenMP if available
if(OpenMP_FOUND)
    target_link_libraries(diskivf PRIVATE OpenMP::OpenMP_CXX)
endif()
