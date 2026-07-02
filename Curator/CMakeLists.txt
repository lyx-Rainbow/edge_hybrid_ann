cmake_minimum_required(VERSION 3.10)
project(curator LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for parallel K-means and batch query
find_package(OpenMP REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Source files
set(CURATOR_SOURCES
    src/cnpy.cpp
    src/kmeans.cpp
    src/cluster_tree.cpp
    src/shortlist.cpp
    src/pq_block_cache.cpp
    src/pq_codec.cpp
    src/flash_store.cpp
    src/temp_index.cpp
    src/complex_predicate.cpp
    src/curator_index.cpp
    src/main.cpp
)

# Header files (for IDE integration, not required for build)
set(CURATOR_HEADERS
    src/cnpy.h
    src/common.h
    src/config.h
    src/bloom_filter.h
    src/distance.h
    src/profiling.h
    src/tree_node.h
    src/kmeans.h
    src/cluster_tree.h
    src/shortlist.h
    src/pq_block_cache.h
    src/pq_codec.h
    src/flash_store.h
    src/temp_index.h
    src/complex_predicate.h
    src/curator_index.h
)

# Executable
add_executable(curator ${CURATOR_SOURCES} ${CURATOR_HEADERS})

# Include directories
target_include_directories(curator PRIVATE src)

# Link libraries
target_link_libraries(curator PRIVATE OpenMP::OpenMP_CXX)

# Optional: FAISS Clustering fallback
# option(CURATOR_USE_FAISS_CLUSTERING "Use FAISS Clustering for K-means" OFF)
# if(CURATOR_USE_FAISS_CLUSTERING)
#     find_package(faiss REQUIRED)
#     target_compile_definitions(curator PRIVATE CURATOR_USE_FAISS_CLUSTERING)
#     target_link_libraries(curator PRIVATE faiss)
# endif()

# Install
install(TARGETS curator DESTINATION bin)
