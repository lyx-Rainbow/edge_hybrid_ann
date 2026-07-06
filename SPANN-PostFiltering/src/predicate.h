// predicate.h — Boolean expression evaluator for PreFiltering / PostFiltering
//
// Supports PREFIX (Polish) notation — matches the algorithm in 2_Utils/predicate.py:
//   "AND 0 1"       → vector has BOTH label 0 AND label 1
//   "OR 0 1"        → vector has EITHER label 0 OR label 1
//   "NOT 0"         → vector does NOT have label 0
//   "AND 0 NOT 1"   → vector has label 0 AND NOT label 1
//
// Uses right-to-left stack evaluation (reversed token iteration).
// Header-only — copy into each index's src/ directory.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace predicate {

// Tokenize a space-delimited formula string.
inline std::vector<std::string> tokenize(const std::string& formula) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : formula) {
        if (c == ' ' || c == '\t') {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        } else {
            current += c;
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// Evaluate a tokenized prefix-notation formula against a label range [begin, end).
// Uses RIGHT-TO-LEFT stack evaluation — matches Python predicate.py evaluate_predicate().
//
// labels must be sorted ascending (for binary_search), but for short lists
// (< 50 labels) linear std::find is used which also works with unsorted lists.
inline bool evaluate(const std::vector<std::string>& tokens,
                     const int32_t* labels_begin,
                     const int32_t* labels_end) {
    std::vector<bool> stack;
    for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
        const std::string& token = *it;
        if (token == "AND") {
            if (stack.size() < 2) return false;
            bool a = stack.back(); stack.pop_back();
            bool b = stack.back(); stack.pop_back();
            stack.push_back(a && b);
        } else if (token == "OR") {
            if (stack.size() < 2) return false;
            bool a = stack.back(); stack.pop_back();
            bool b = stack.back(); stack.pop_back();
            stack.push_back(a || b);
        } else if (token == "NOT") {
            if (stack.empty()) return false;
            bool a = stack.back(); stack.pop_back();
            stack.push_back(!a);
        } else {
            // Variable (label ID): check membership via linear search.
            // Label lists per vector are typically short (< 50), so
            // linear search is competitive with binary_search.
            int32_t label_id = std::stoi(token);
            stack.push_back(std::find(labels_begin, labels_end, label_id)
                            != labels_end);
        }
    }
    return stack.size() == 1 && stack.back();
}

// Convenience overload: tokenize + evaluate.
inline bool evaluate(const std::string& formula,
                     const int32_t* labels_begin,
                     const int32_t* labels_end) {
    return evaluate(tokenize(formula), labels_begin, labels_end);
}

} // namespace predicate
