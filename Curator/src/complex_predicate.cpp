// complex_predicate.cpp — Predicate expression implementation
#include "complex_predicate.h"

#include <iostream>
#include <sstream>
#include <stack>

#include "common.h"

namespace curator {
namespace predicate {

// ============================================================================
// StateNode
// ============================================================================
StateNode::StateNode(Type type, const Buffer& short_list, const Buffer& exclude_list)
    : type(type), short_list(short_list), exclude_list(exclude_list) {}

StateNode::StateNode(Type type, Buffer&& short_list, Buffer&& exclude_list)
    : type(type),
      short_list(std::move(short_list)),
      exclude_list(std::move(exclude_list)) {}

std::string StateNode::type_to_str() const {
    switch (type) {
        case Type::NONE:    return "NONE";
        case Type::SOME:    return "SOME";
        case Type::MOST:    return "MOST";
        case Type::ALL:     return "ALL";
        default:            return "UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, const StateNode& state) {
    os << "State(Type: " << state.type_to_str() << ", Short List: [";
    for (size_t i = 0; i < state.short_list.size(); ++i) {
        os << state.short_list[i] << (i < state.short_list.size() - 1 ? ", " : "");
    }
    os << "], Exclude List: [";
    for (size_t i = 0; i < state.exclude_list.size(); ++i) {
        os << state.exclude_list[i] << (i < state.exclude_list.size() - 1 ? ", " : "");
    }
    os << "])";
    return os;
}

// ============================================================================
// make_state
// ============================================================================
State make_state(Type type, bool concretize, const Buffer& list) {
    switch (type) {
        case Type::SOME: {
            if (!concretize) {
                CURATOR_THROW_MSG("Should not manually create SOME states in abstract mode");
            } else if (list.empty()) {
                return STATE_NONE;
            } else {
                return std::make_shared<StateNode>(type, list, EMPTY_BUFFER);
            }
        }
        case Type::MOST: {
            if (!concretize) {
                CURATOR_THROW_MSG("Should not manually create MOST states in abstract mode");
            } else if (list.empty()) {
                return STATE_ALL;
            } else {
                return std::make_shared<StateNode>(type, EMPTY_BUFFER, list);
            }
        }
        default:
            CURATOR_THROW_MSG("Should not manually create states of type ALL, NONE, or UNKNOWN");
    }
}

// ============================================================================
// VarMapNode
// ============================================================================
VarMapNode::VarMapNode(std::unordered_map<std::string, State>&& var_map)
    : var_map(std::move(var_map)) {}

VarMap VarMapNode::update(const std::string& name, State new_state) const {
    VarMap new_var_map = std::make_shared<VarMapNode>(*this);
    new_var_map->var_map[name] = new_state;
    return new_var_map;
}

VarMap VarMapNode::update(const std::unordered_map<std::string, State>& new_var_map) const {
    VarMap updated_var_map = std::make_shared<VarMapNode>(*this);
    for (auto& [name, state] : new_var_map) {
        updated_var_map->var_map[name] = state;
    }
    return updated_var_map;
}

const State& VarMapNode::get(const std::string& name) const {
    if (var_map.find(name) == var_map.end()) {
        return STATE_UNKNOWN;
    }
    return var_map.at(name);
}

std::vector<std::string> VarMapNode::unresolved_vars() const {
    std::vector<std::string> unresolved;
    for (const auto& [name, state] : var_map) {
        if (state->type == Type::UNKNOWN) {
            unresolved.push_back(name);
        }
    }
    return unresolved;
}

std::ostream& operator<<(std::ostream& os, const VarMapNode& var_map) {
    os << "VarMap {";
    for (const auto& [name, state] : var_map.var_map) {
        os << "\n  " << name << " → " << *state;
    }
    os << "\n}";
    return os;
}

// ============================================================================
// ExprNode evaluations
// ============================================================================
State VariableNode::evaluate(VarMap var_map, bool concretize) const {
    return var_map->get(var_name);
}

State NotNode::evaluate(VarMap var_map, bool concretize) const {
    auto op_state = operand->evaluate(var_map, concretize);
    switch (op_state->type) {
        case Type::NONE:    return STATE_ALL;
        case Type::ALL:     return STATE_NONE;
        case Type::SOME:    return make_state(Type::MOST, concretize, op_state->short_list);
        case Type::MOST:    return make_state(Type::SOME, concretize, op_state->exclude_list);
        default:            return STATE_UNKNOWN;
    }
}

State AndNode::evaluate(VarMap var_map, bool concretize) const {
    auto l = left->evaluate(var_map, concretize);
    auto r = right->evaluate(var_map, concretize);

    if (l->type == Type::NONE || r->type == Type::NONE) return STATE_NONE;

    if (l->type == Type::ALL && r->type == Type::ALL) return STATE_ALL;

    if ((l->type == Type::ALL || l->type == Type::MOST) &&
        (r->type == Type::ALL || r->type == Type::MOST)) {
        return make_state(Type::MOST, concretize,
                          buffer_union(l->exclude_list, r->exclude_list));
    }

    if ((l->type == Type::ALL || l->type == Type::MOST) &&
        (r->type == Type::SOME)) {
        return make_state(Type::SOME, concretize,
                          buffer_difference(r->short_list, l->exclude_list));
    }

    if ((l->type == Type::SOME) &&
        (r->type == Type::ALL || r->type == Type::MOST)) {
        return make_state(Type::SOME, concretize,
                          buffer_difference(l->short_list, r->exclude_list));
    }

    if (l->type == Type::SOME && r->type == Type::SOME) {
        return make_state(Type::SOME, concretize,
                          buffer_intersect(l->short_list, r->short_list));
    }

    return STATE_UNKNOWN;
}

State OrNode::evaluate(VarMap var_map, bool concretize) const {
    auto l = left->evaluate(var_map, concretize);
    auto r = right->evaluate(var_map, concretize);

    if (l->type == Type::ALL || r->type == Type::ALL) return STATE_ALL;

    if (l->type == Type::NONE && r->type == Type::NONE) return STATE_NONE;

    if ((l->type == Type::NONE || l->type == Type::SOME) &&
        (r->type == Type::NONE || r->type == Type::SOME)) {
        return make_state(Type::SOME, concretize,
                          buffer_union(l->short_list, r->short_list));
    }

    if ((l->type == Type::MOST || l->type == Type::ALL) ||
        (r->type == Type::MOST || r->type == Type::ALL)) {
        return STATE_ALL;
    }

    return STATE_UNKNOWN;
}

// ============================================================================
// Tokenization (Polish Notation)
// ============================================================================
std::vector<std::string> tokenize_formula(const std::string& formula) {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : formula) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else if (c == '(' || c == ')') {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            tokens.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

// ============================================================================
// Parsing: Polish Notation → Expr AST (Shunting-yard style)
// ============================================================================
Expr parse_formula(
        const std::string& formula,
        std::unordered_map<std::string, State>* var_map) {
    auto tokens = tokenize_formula(formula);

    // Reverse Polish Notation evaluation to build AST
    std::stack<Expr> expr_stack;

    for (const auto& token : tokens) {
        if (token == "AND" || token == "&" || token == "∧") {
            if (expr_stack.size() < 2) {
                CURATOR_THROW_FMT("Invalid formula: not enough operands for AND (token: %s)", token.c_str());
            }
            auto right = std::move(expr_stack.top()); expr_stack.pop();
            auto left = std::move(expr_stack.top()); expr_stack.pop();
            expr_stack.push(make_and(std::move(left), std::move(right)));
        } else if (token == "OR" || token == "|" || token == "∨") {
            if (expr_stack.size() < 2) {
                CURATOR_THROW_FMT("Invalid formula: not enough operands for OR (token: %s)", token.c_str());
            }
            auto right = std::move(expr_stack.top()); expr_stack.pop();
            auto left = std::move(expr_stack.top()); expr_stack.pop();
            expr_stack.push(make_or(std::move(left), std::move(right)));
        } else if (token == "NOT" || token == "!" || token == "¬") {
            if (expr_stack.empty()) {
                CURATOR_THROW_FMT("Invalid formula: not enough operands for NOT (token: %s)", token.c_str());
            }
            auto operand = std::move(expr_stack.top()); expr_stack.pop();
            expr_stack.push(make_not(std::move(operand)));
        } else if (token == "(" || token == ")") {
            // Skip parentheses (RPN doesn't need them, but tolerate them)
            continue;
        } else {
            // Variable
            expr_stack.push(make_var(token));
            if (var_map != nullptr) {
                (*var_map)[token] = STATE_UNKNOWN;
            }
        }
    }

    if (expr_stack.size() != 1) {
        CURATOR_THROW_FMT("Invalid formula: %zu leftover expressions on stack (expected 1)",
                          expr_stack.size());
    }

    return std::move(expr_stack.top());
}

} // namespace predicate
} // namespace curator
