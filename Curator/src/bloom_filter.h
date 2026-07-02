// bloom_filter.h — Bloom Filter for Curator index (namespace curator)
// Adapted from original BloomFilter.h, removing compressible_bloom_filter
// Original: Arash Partow - 2000, MIT License
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace curator {

static const std::size_t bits_per_char = 0x08;
static const unsigned char bit_mask[bits_per_char] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
};

class bloom_parameters {
public:
    bloom_parameters()
        : minimum_size(1),
          maximum_size(std::numeric_limits<unsigned long long int>::max()),
          minimum_number_of_hashes(1),
          maximum_number_of_hashes(std::numeric_limits<unsigned int>::max()),
          projected_element_count(10000),
          false_positive_probability(1.0 / projected_element_count),
          random_seed(0xA5A5A5A55A5A5A5AULL) {}

    virtual ~bloom_parameters() {}

    inline bool operator!() {
        return (minimum_size > maximum_size) ||
               (minimum_number_of_hashes > maximum_number_of_hashes) ||
               (minimum_number_of_hashes < 1) ||
               (0 == maximum_number_of_hashes) ||
               (0 == projected_element_count) ||
               (false_positive_probability < 0.0) ||
               (std::numeric_limits<double>::infinity() == std::abs(false_positive_probability)) ||
               (0 == random_seed) ||
               (0xFFFFFFFFFFFFFFFFULL == random_seed);
    }

    bool compute_optimal_parameters() {
        if (!(*this))
            return false;

        double min_m = std::numeric_limits<double>::infinity();
        double min_k = 0.0;
        double k = 1.0;

        while (k < 1000.0) {
            const double numerator = (-k * projected_element_count);
            const double denominator = std::log(1.0 - std::pow(false_positive_probability, 1.0 / k));
            const double curr_m = numerator / denominator;
            if (curr_m < min_m) {
                min_m = curr_m;
                min_k = k;
            }
            k += 1.0;
        }

        optimal_parameters_t& optp = optimal_parameters;

        optp.number_of_hashes = static_cast<unsigned int>(min_k);
        optp.table_size = static_cast<unsigned long long int>(min_m);
        optp.table_size += (((optp.table_size % bits_per_char) != 0)
            ? (bits_per_char - (optp.table_size % bits_per_char)) : 0);

        if (optp.number_of_hashes < minimum_number_of_hashes)
            optp.number_of_hashes = minimum_number_of_hashes;
        else if (optp.number_of_hashes > maximum_number_of_hashes)
            optp.number_of_hashes = maximum_number_of_hashes;

        if (optp.table_size < minimum_size)
            optp.table_size = minimum_size;
        else if (optp.table_size > maximum_size)
            optp.table_size = maximum_size;

        return true;
    }

    unsigned long long int minimum_size;
    unsigned long long int maximum_size;
    unsigned int minimum_number_of_hashes;
    unsigned int maximum_number_of_hashes;
    unsigned long long int projected_element_count;
    double false_positive_probability;
    unsigned long long int random_seed;

    struct optimal_parameters_t {
        optimal_parameters_t() : number_of_hashes(0), table_size(0) {}
        unsigned int number_of_hashes;
        unsigned long long int table_size;
    };

    optimal_parameters_t optimal_parameters;

    unsigned int& number_of_hashes = optimal_parameters.number_of_hashes;
    unsigned long long int& table_size_ = optimal_parameters.table_size;
};

class bloom_filter {
public:
    bloom_filter() : salt_count_(0), table_size_(0), projected_element_count_(0),
                     inserted_element_count_(0), random_seed_(0), desired_false_positive_probability_(0.0) {}

    bloom_filter(const bloom_parameters& p)
        : projected_element_count_(p.projected_element_count),
          inserted_element_count_(0),
          random_seed_((p.random_seed * 0xA5A5A5A5 + 1) % std::numeric_limits<unsigned long long int>::max()),
          desired_false_positive_probability_(p.false_positive_probability) {
        salt_count_ = p.optimal_parameters.number_of_hashes;
        table_size_ = p.optimal_parameters.table_size;
        generate_unique_salt();
        bit_table_.resize(table_size_ / bits_per_char, static_cast<unsigned char>(0x00));
    }

    bloom_filter(const bloom_filter& filter) {
        this->operator=(filter);
    }

    bloom_filter& operator=(const bloom_filter& filter) {
        if (this != &filter) {
            salt_count_ = filter.salt_count_;
            table_size_ = filter.table_size_;
            bit_table_ = filter.bit_table_;
            salt_ = filter.salt_;
            projected_element_count_ = filter.projected_element_count_;
            inserted_element_count_ = filter.inserted_element_count_;
            random_seed_ = filter.random_seed_;
            desired_false_positive_probability_ = filter.desired_false_positive_probability_;
        }
        return *this;
    }

    inline void insert(unsigned long long int key) {
        for (std::size_t i = 0; i < salt_.size(); ++i) {
            std::size_t bit = hash_ap(key, salt_[i]) % table_size_;
            bit_table_[bit / bits_per_char] |= bit_mask[bit % bits_per_char];
        }
        ++inserted_element_count_;
    }

    template <typename InputIterator>
    inline void insert(const InputIterator begin, const InputIterator end) {
        for (InputIterator it = begin; it != end; ++it) {
            insert(*it);
        }
    }

    inline bool contains(unsigned long long int key) const {
        for (std::size_t i = 0; i < salt_.size(); ++i) {
            std::size_t bit = hash_ap(key, salt_[i]) % table_size_;
            if ((bit_table_[bit / bits_per_char] & bit_mask[bit % bits_per_char]) != bit_mask[bit % bits_per_char]) {
                return false;
            }
        }
        return true;
    }

    inline void clear() {
        std::fill(bit_table_.begin(), bit_table_.end(), static_cast<unsigned char>(0x00));
        inserted_element_count_ = 0;
    }

    bloom_filter& operator&=(const bloom_filter& filter) {
        if (this != &filter) {
            if (table_size_ != filter.table_size_ || salt_count_ != filter.salt_count_) {
                return *this;
            }
            for (std::size_t i = 0; i < bit_table_.size(); ++i) {
                bit_table_[i] &= filter.bit_table_[i];
            }
        }
        return *this;
    }

    bloom_filter& operator|=(const bloom_filter& filter) {
        if (this != &filter) {
            if (table_size_ != filter.table_size_ || salt_count_ != filter.salt_count_) {
                return *this;
            }
            for (std::size_t i = 0; i < bit_table_.size(); ++i) {
                bit_table_[i] |= filter.bit_table_[i];
            }
        }
        return *this;
    }

    bloom_filter& operator^=(const bloom_filter& filter) {
        if (this != &filter) {
            if (table_size_ != filter.table_size_ || salt_count_ != filter.salt_count_) {
                return *this;
            }
            for (std::size_t i = 0; i < bit_table_.size(); ++i) {
                bit_table_[i] ^= filter.bit_table_[i];
            }
        }
        return *this;
    }

    inline bloom_filter operator|(const bloom_filter& f) const {
        bloom_filter result = *this;
        result |= f;
        return result;
    }

    inline bloom_filter operator&(const bloom_filter& f) const {
        bloom_filter result = *this;
        result &= f;
        return result;
    }

    inline std::size_t size() const { return bit_table_.size(); }
    inline unsigned long long int element_count() const { return inserted_element_count_; }
    inline unsigned int hash_count() const { return static_cast<unsigned int>(salt_.size()); }

private:
    inline std::size_t hash_ap(unsigned long long int key, unsigned long long int salt) const {
        unsigned long long int hash = key ^ salt;
        hash = ~hash + (hash << 21);
        hash = hash ^ (hash >> 24);
        hash = (hash + (hash << 3)) + (hash << 8);
        hash = hash ^ (hash >> 14);
        hash = (hash + (hash << 2)) + (hash << 4);
        hash = hash ^ (hash >> 28);
        hash = hash + (hash << 31);
        return static_cast<std::size_t>(hash);
    }

    void generate_unique_salt() {
        const unsigned int predef_salt_count = 128;
        static const unsigned long long int predef_salt[predef_salt_count] = {
            0xAAAAAAAA55555555ULL, 0x3333333399999999ULL, 0xCCCCCCCC33333333ULL,
            0x6666666699999999ULL, 0x99999999CCCCCCCCULL, 0x3333333366666666ULL,
            0xCCCCCCCC99999999ULL, 0x6666666633333333ULL, 0x9999999900000000ULL,
            0x0000000099999999ULL, 0xAAAAAAAA33333333ULL, 0x55555555CCCCCCCCULL,
            0x33333333AAAAAAAAULL, 0xCCCCCCCC55555555ULL, 0x66666666AAAAAAAAULL,
            0x9999999955555555ULL, 0xAAAAAAAA66666666ULL, 0x5555555599999999ULL,
            0x3333333300000000ULL, 0xCCCCCCCC00000000ULL, 0x6666666600000000ULL,
            0x99999999FFFFFFFFULL, 0xAAAAAAAAFFFFFFFFULL, 0x55555555FFFFFFFFULL,
            0x33333333FFFFFFFFULL, 0xCCCCCCCCFFFFFFFFULL, 0x66666666FFFFFFFFULL,
            0x0000000033333333ULL, 0xFFFFFFFF33333333ULL, 0x00000000CCCCCCCCULL,
            0xFFFFFFFFCCCCCCCCULL, 0x0000000066666666ULL, 0xFFFFFFFF66666666ULL,
            0x0000000099999999ULL, 0x00000000AAAAAAAAULL, 0xFFFFFFFFAAAAAAAAULL,
            0x0000000055555555ULL, 0xFFFFFFFF55555555ULL, 0x0000000000000000ULL,
            0xFFFFFFFFFFFFFFFFULL, 0x1111111111111111ULL, 0x2222222222222222ULL,
            0x4444444444444444ULL, 0x8888888888888888ULL, 0x7777777777777777ULL,
            0xBBBBBBBBBBBBBBBBULL, 0xDDDDDDDDDDDDDDDDULL, 0xEEEEEEEEEEEEEEEEULL,
            0xA5A5A5A5A5A5A5A5ULL, 0x5A5A5A5A5A5A5A5AULL, 0x3C3C3C3C3C3C3C3CULL,
            0xC3C3C3C3C3C3C3C3ULL, 0x6969696969696969ULL, 0x9696969696969696ULL,
            0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL, 0xAAAAAAAA55555556ULL,
            0x333333339999999AULL, 0xCCCCCCCC33333334ULL, 0x666666669999999BULL,
            0x99999999CCCCCCCDULL, 0x3333333366666667ULL, 0xCCCCCCCC9999999BULL,
            0x6666666633333334ULL, 0x9999999900000001ULL, 0x000000009999999AULL,
            0xAAAAAAAA33333334ULL, 0x55555555CCCCCCCDULL, 0x33333333AAAAAAABULL,
            0xCCCCCCCC55555556ULL, 0x66666666AAAAAAABULL, 0x9999999955555556ULL,
            0xAAAAAAAA66666667ULL, 0x555555559999999BULL, 0x3333333300000001ULL,
            0xCCCCCCCC00000001ULL, 0x6666666600000001ULL, 0x9999999900000000ULL,
            0xAAAAAAA9FFFFFFFFULL, 0x5555555400000000ULL, 0x33333332FFFFFFFFULL,
            0xCCCCCCCBFFFFFFFFULL, 0x66666665FFFFFFFFULL, 0x0000000033333334ULL,
            0xFFFFFFFF33333334ULL, 0x00000000CCCCCCCDULL, 0xFFFFFFFFCCCCCCCDULL,
            0x0000000066666667ULL, 0xFFFFFFFF66666667ULL, 0x000000009999999BULL,
            0x00000000AAAAAAABULL, 0xFFFFFFFFAAAAAAABULL, 0x0000000055555556ULL,
            0xFFFFFFFF55555556ULL
        };

        if (salt_count_ <= predef_salt_count) {
            salt_.assign(predef_salt, predef_salt + salt_count_);
        } else {
            salt_.assign(predef_salt, predef_salt + predef_salt_count);
            for (std::size_t i = predef_salt_count; i < salt_count_; ++i) {
                salt_.push_back(predef_salt[i % predef_salt_count] ^ random_seed_);
            }
        }
    }

    std::vector<unsigned char> bit_table_;
    unsigned int salt_count_;
    unsigned long long int table_size_;
    unsigned long long int projected_element_count_;
    unsigned long long int inserted_element_count_;
    unsigned long long int random_seed_;
    double desired_false_positive_probability_;
    std::vector<unsigned long long int> salt_;
};

} // namespace curator
