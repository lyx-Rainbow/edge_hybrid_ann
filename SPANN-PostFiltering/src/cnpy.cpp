// cnpy.cpp — minimal NumPy .npy file reader implementation
#include "cnpy.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cnpy {

namespace {
inline void check_dtype(const std::string& dtype_str, char expected) {
    if (dtype_str.size() < 2 || dtype_str[1] != expected) {
        std::string msg = "cnpy: dtype mismatch. File has '";
        msg += dtype_str;
        msg += "', expected '";
        msg += expected;
        msg += "'";
        throw std::runtime_error(msg);
    }
}
} // anonymous namespace

bool parse_npy_header(
        FILE* fp,
        std::string& dtype_str,
        bool& fortran_order,
        std::vector<size_t>& shape,
        size_t& data_offset) {
    char magic[7] = {0};
    if (std::fread(magic, 1, 6, fp) != 6) return false;
    if (std::string(magic, 6) != "\x93NUMPY") return false;

    uint8_t major = 0, minor = 0;
    if (std::fread(&major, 1, 1, fp) != 1) return false;
    if (std::fread(&minor, 1, 1, fp) != 1) return false;

    uint16_t header_len = 0;
    if (major == 1) {
        if (std::fread(&header_len, 2, 1, fp) != 1) return false;
    } else if (major == 2 || major == 3) {
        uint32_t hlen32 = 0;
        if (std::fread(&hlen32, 4, 1, fp) != 1) return false;
        header_len = static_cast<uint16_t>(hlen32);
    } else {
        return false;
    }

    data_offset = 6 + 2 + (major == 1 ? 2 : 4) + header_len;

    std::vector<char> header_buf(header_len + 1, 0);
    if (std::fread(header_buf.data(), 1, header_len, fp) != header_len) return false;
    std::string header_str(header_buf.data(), header_len);

    // Parse 'descr'
    auto descr_pos = header_str.find("'descr'");
    if (descr_pos == std::string::npos) return false;
    auto descr_start = header_str.find('\'', descr_pos + 7);
    if (descr_start == std::string::npos) return false;
    auto descr_end = header_str.find('\'', descr_start + 1);
    if (descr_end == std::string::npos) return false;
    dtype_str = header_str.substr(descr_start + 1, descr_end - descr_start - 1);

    // Parse 'fortran_order'
    fortran_order = (header_str.find("'fortran_order': True") != std::string::npos);

    // Parse 'shape'
    auto shape_pos = header_str.find("'shape'");
    if (shape_pos == std::string::npos) return false;
    auto shape_start = header_str.find('(', shape_pos);
    if (shape_start == std::string::npos) return false;
    auto shape_end = header_str.find(')', shape_start);
    if (shape_end == std::string::npos) return false;
    std::string shape_str = header_str.substr(shape_start + 1, shape_end - shape_start - 1);

    shape.clear();
    if (!shape_str.empty()) {
        size_t pos = 0;
        while (pos < shape_str.size()) {
            while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ','))
                pos++;
            if (pos >= shape_str.size()) break;
            size_t end = pos;
            while (end < shape_str.size() && shape_str[end] >= '0' && shape_str[end] <= '9')
                end++;
            if (end > pos) {
                shape.push_back(std::stoull(shape_str.substr(pos, end - pos)));
            }
            pos = end;
        }
    }

    return true;
}

// ALL 2-parameter specializations FIRST (to avoid "after instantiation" errors)
template <>
std::vector<float> npy_load<float>(const std::string& fname, std::vector<size_t>& shape) {
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("cnpy: cannot open file: " + fname);
    std::string dtype_str; bool fortran_order; size_t data_offset;
    if (!parse_npy_header(fp, dtype_str, fortran_order, shape, data_offset)) {
        std::fclose(fp); throw std::runtime_error("cnpy: failed to parse header: " + fname);
    }
    check_dtype(dtype_str, 'f');
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    std::vector<float> data(n_elements);
    std::fread(data.data(), sizeof(float), n_elements, fp);
    std::fclose(fp);
    return data;
}

template <>
std::vector<int32_t> npy_load<int32_t>(const std::string& fname, std::vector<size_t>& shape) {
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("cnpy: cannot open file: " + fname);
    std::string dtype_str; bool fortran_order; size_t data_offset;
    if (!parse_npy_header(fp, dtype_str, fortran_order, shape, data_offset)) {
        std::fclose(fp); throw std::runtime_error("cnpy: failed to parse header: " + fname);
    }
    check_dtype(dtype_str, 'i');
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    std::vector<int32_t> data(n_elements);
    std::fread(data.data(), sizeof(int32_t), n_elements, fp);
    std::fclose(fp);
    return data;
}

template <>
std::vector<int64_t> npy_load<int64_t>(const std::string& fname, std::vector<size_t>& shape) {
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("cnpy: cannot open file: " + fname);
    std::string dtype_str; bool fortran_order; size_t data_offset;
    if (!parse_npy_header(fp, dtype_str, fortran_order, shape, data_offset)) {
        std::fclose(fp); throw std::runtime_error("cnpy: failed to parse header: " + fname);
    }
    check_dtype(dtype_str, 'l');
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    std::vector<int64_t> data(n_elements);
    std::fread(data.data(), sizeof(int64_t), n_elements, fp);
    std::fclose(fp);
    return data;
}

template <>
std::vector<uint8_t> npy_load<uint8_t>(const std::string& fname, std::vector<size_t>& shape) {
    FILE* fp = std::fopen(fname.c_str(), "rb");
    if (!fp) throw std::runtime_error("cnpy: cannot open file: " + fname);
    std::string dtype_str; bool fortran_order; size_t data_offset;
    if (!parse_npy_header(fp, dtype_str, fortran_order, shape, data_offset)) {
        std::fclose(fp); throw std::runtime_error("cnpy: failed to parse header: " + fname);
    }
    check_dtype(dtype_str, 'u');
    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    std::vector<uint8_t> data(n_elements);
    std::fread(data.data(), sizeof(uint8_t), n_elements, fp);
    std::fclose(fp);
    return data;
}

// THEN 1-parameter specializations (delegate to 2-param)
template <>
std::vector<float> npy_load<float>(const std::string& fname) {
    std::vector<size_t> shape;
    return npy_load<float>(fname, shape);
}

template <>
std::vector<int32_t> npy_load<int32_t>(const std::string& fname) {
    std::vector<size_t> shape;
    return npy_load<int32_t>(fname, shape);
}

template <>
std::vector<int64_t> npy_load<int64_t>(const std::string& fname) {
    std::vector<size_t> shape;
    return npy_load<int64_t>(fname, shape);
}

template <>
std::vector<uint8_t> npy_load<uint8_t>(const std::string& fname) {
    std::vector<size_t> shape;
    return npy_load<uint8_t>(fname, shape);
}

} // namespace cnpy
