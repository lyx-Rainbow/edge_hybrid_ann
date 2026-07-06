// cnpy.h — minimal NumPy .npy file reader for C++
// Based on https://github.com/rogersce/cnpy (MIT license)
// Supports: float32, int32, int64, uint8
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cnpy {

// Return the in-memory data from a .npy file.
// T must match the dtype on disk.
template <typename T>
std::vector<T> npy_load(const std::string& fname);

// Expected shape is filled in by the function.
template <typename T>
std::vector<T> npy_load(const std::string& fname, std::vector<size_t>& shape);

// Parse the .npy header, returning dtype string, fortran_order flag, and shape.
// data_offset is set to the byte position where raw data begins.
bool parse_npy_header(
        FILE* fp,
        std::string& dtype_str,
        bool& fortran_order,
        std::vector<size_t>& shape,
        size_t& data_offset);

} // namespace cnpy
