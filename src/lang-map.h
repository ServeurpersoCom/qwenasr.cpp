#pragma once
// lang-map.h: forced language resolution. Mirrors qwen_asr
// normalize_language_name and validate_language: normalize to canonical casing
// then accept only the supported names. The prompt receives the canonical name
// itself, empty leaves the model to detect the language.

#include <cctype>
#include <string>
#include <unordered_set>

static const std::unordered_set<std::string> SUPPORTED_LANGUAGES = {
    "Chinese",    "English", "Cantonese", "Arabic",   "German",    "French",     "Spanish",  "Portuguese",
    "Indonesian", "Italian", "Korean",    "Russian",  "Thai",      "Vietnamese", "Japanese", "Turkish",
    "Hindi",      "Malay",   "Dutch",     "Swedish",  "Danish",    "Finnish",    "Polish",   "Czech",
    "Filipino",   "Persian", "Greek",     "Romanian", "Hungarian", "Macedonian",
};

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
