cmake_minimum_required(VERSION 3.12)
project(spann_pf LANGUAGES CXX)

# C++17 required (SPTAG also uses C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for batch_query
find_package(OpenMP)

# SPTAG dependencies
find_package(Boost 1.66 COMPONENTS system thread serialization filesystem REQUIRED)
find_package(TBB REQUIRED)

# Compiler flags (consistent with SPTAG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# ★ SPTAG library paths (SPTAG-main must be compiled first)
set(SPTAG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/SPTAG-main)
set(SPTAG_BUILD_DIR ${SPTAG_ROOT}/Release)

# Include paths
include_directories(${SPTAG_ROOT}/AnnService)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})

# Source files
set(SPANN_SOURCES
    src/cnpy.cpp
    src/spann_pf_index.cpp
    src/main.cpp
)

# Header files (for IDE integration, not required for build)
set(SPANN_HEADERS
    src/cnpy.h
    src/config.h
    src/distance.h
    src/predicate.h
    src/spann_pf_index.h
)

# Executable
add_executable(spann_pf ${SPANN_SOURCES} ${SPANN_HEADERS})

# Include search paths
target_include_directories(spann_pf PRIVATE src)

# Link SPTAG static library and all its dependencies
target_link_libraries(spann_pf PRIVATE
    ${SPTAG_BUILD_DIR}/libSPTAGLibStatic.a
    ${SPTAG_BUILD_DIR}/libDistanceUtils.a
    ${Boost_LIBRARIES}
    tbb
    zstd
    -lrt
    -lpthread
)

# Link OpenMP if available
if(OpenMP_FOUND)
    target_link_libraries(spann_pf PRIVATE OpenMP::OpenMP_CXX)
endif()
