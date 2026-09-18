#pragma once
// lang-map.h: forced language resolution. Mirrors qwen_asr
// normalize_language_name and validate_language: normalize to canonical casing
// then accept only the supported names. The prompt receives the canonical name
// itself, empty leaves the model to detect the language.

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

static const std::unordered_set<std::string> SUPPORTED_LANGUAGES = {
    "Chinese",    "English", "Cantonese", "Arabic",   "German",    "French",     "Spanish",  "Portuguese",
    "Indonesian", "Italian", "Korean",    "Russian",  "Thai",      "Vietnamese", "Japanese", "Turkish",
    "Hindi",      "Malay",   "Dutch",     "Swedish",  "Danish",    "Finnish",    "Polish",   "Czech",
    "Filipino",   "Persian", "Greek",     "Romanian", "Hungarian", "Macedonian",
};

// Sorted view of the same set, for the enumeration the ABI exposes: an
// unordered_set has no stable order, and a selector that reshuffles between
// two runs is unusable.
static inline const std::vector<std::string> & supported_languages() {
    static const std::vector<std::string> sorted = [] {
        std::vector<std::string> out(SUPPORTED_LANGUAGES.begin(), SUPPORTED_LANGUAGES.end());
        std::sort(out.begin(), out.end());
        return out;
    }();
    return sorted;
}

// Canonical casing: first byte upper, the rest lower. The fold is ASCII only,
// multibyte UTF-8 bytes stay >= 0x80 and pass through untouched, so unsupported
// non-ASCII input simply fails the lookup. Returns the supported canonical
// name, or empty for "auto", "none", and anything unsupported, which detects.
static inline std::string resolve_language(const std::string & language) {
    if (language.empty()) {
        return "";
    }
    std::string s = language;
    s[0]          = (char) std::toupper((unsigned char) s[0]);
    for (size_t i = 1; i < s.size(); i++) {
        s[i] = (char) std::tolower((unsigned char) s[i]);
    }
    return SUPPORTED_LANGUAGES.count(s) ? s : "";
}
