/**
 * @file json_lite.h
 * @brief Minimal strict JSON reader for the CLI's structured inputs
 *        (--node-locks, --bet-sizing).
 *
 * 2026-10-06 audit: the node-lock reader was a substring scanner keyed on
 * `"history":"` — a space after the colon (Python's json.dumps default) made
 * it read an empty history (the lock landed on the ROOT) or an empty combo
 * (the lock was silently dropped). This parses real JSON and reports the
 * position of anything it does not understand.
 */

#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace deepsolver::json {

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::pair<std::string, Value>> object;   // insertion order

    bool is_object() const { return kind == Kind::Object; }
    bool is_array()  const { return kind == Kind::Array; }
    bool is_string() const { return kind == Kind::String; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_bool()   const { return kind == Kind::Bool; }

    /// Member lookup; nullptr when absent (or when this is not an object).
    const Value* find(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        for (const auto& kv : object) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value parse_document() {
        Value v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("invalid JSON (") + what +
                                 ") at offset " + std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool consume(char c) {
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == c) { ++pos_; return true; }
        return false;
    }

    void expect(char c, const char* what) {
        if (!consume(c)) fail(what);
    }

    bool match_word(const char* w) {
        std::size_t n = 0;
        while (w[n] != '\0') ++n;
        if (s_.compare(pos_, n, w) == 0) { pos_ += n; return true; }
        return false;
    }

    Value parse_value(int depth) {
        if (depth > 64) fail("nesting too deep");
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end");
        const char c = s_[pos_];
        Value v;
        if (c == '{') {
            ++pos_;
            v.kind = Value::Kind::Object;
            if (consume('}')) return v;
            do {
                skip_ws();
                if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a key");
                std::string key = parse_string();
                expect(':', "expected ':'");
                v.object.emplace_back(std::move(key), parse_value(depth + 1));
            } while (consume(','));
            expect('}', "expected ',' or '}'");
        } else if (c == '[') {
            ++pos_;
            v.kind = Value::Kind::Array;
            if (consume(']')) return v;
            do {
                v.array.push_back(parse_value(depth + 1));
            } while (consume(','));
            expect(']', "expected ',' or ']'");
        } else if (c == '"') {
            v.kind = Value::Kind::String;
            v.string = parse_string();
        } else if (match_word("true")) {
            v.kind = Value::Kind::Bool;
            v.boolean = true;
        } else if (match_word("false")) {
            v.kind = Value::Kind::Bool;
        } else if (match_word("null")) {
            v.kind = Value::Kind::Null;
        } else {
            v.kind = Value::Kind::Number;
            v.number = parse_number();
        }
        return v;
    }

    std::string parse_string() {
        ++pos_;   // opening quote
        std::string out;
        while (pos_ < s_.size()) {
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (pos_ >= s_.size()) break;
            const char e = s_[pos_++];
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("short \\u escape");
                    const unsigned long cp =
                        std::strtoul(s_.substr(pos_, 4).c_str(), nullptr, 16);
                    pos_ += 4;
                    // The CLI inputs are ASCII; anything else is kept as '?'.
                    out += (cp < 0x80) ? static_cast<char>(cp) : '?';
                    break;
                }
                default: fail("bad escape");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        const std::size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size() &&
               ((s_[pos_] >= '0' && s_[pos_] <= '9') || s_[pos_] == '.' ||
                s_[pos_] == 'e' || s_[pos_] == 'E' || s_[pos_] == '-' || s_[pos_] == '+')) {
            ++pos_;
        }
        if (pos_ == start) fail("unexpected character");
        const std::string tok = s_.substr(start, pos_ - start);
        char* end = nullptr;
        const double d = std::strtod(tok.c_str(), &end);
        if (end == nullptr || *end != '\0') fail("bad number");
        return d;
    }
};

}  // namespace detail

/// Parse `text` as one JSON document; throws std::runtime_error on any error.
inline Value parse(const std::string& text) {
    return detail::Parser(text).parse_document();
}

}  // namespace deepsolver::json
