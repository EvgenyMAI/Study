#include "tokenizer.h"
#include <iostream>
#include <algorithm>

Tokenizer::Tokenizer() 
    : lowercase_(true)
    , remove_punctuation_(true)
    , min_token_length_(1)
{
    // Regex не нужен, но оставим для совместимости интерфейса
    word_pattern_ = std::regex("");
    punctuation_pattern_ = std::regex("");
}

// Проверка, является ли символ буквой (ASCII латиница)
inline bool is_ascii_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Проверка, является ли символ цифрой
inline bool is_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

// Проверка, является ли последовательность байтов кириллицей
inline bool is_cyrillic_start(unsigned char c1, unsigned char c2) {
    // Кириллица: 0xD0 0x80-0xBF и 0xD1 0x80-0xBF
    return ((c1 == 0xD0 && c2 >= 0x80 && c2 <= 0xBF) ||
            (c1 == 0xD1 && c2 >= 0x80 && c2 <= 0xBF));
}

// Проверка, может ли символ быть частью токена
inline bool is_token_char(const std::string& text, size_t pos, size_t& advance) {
    if (pos >= text.size()) {
        advance = 0;
        return false;
    }
    
    unsigned char c = text[pos];
    
    // ASCII буква
    if (is_ascii_letter(c)) {
        advance = 1;
        return true;
    }
    
    // Цифра
    if (is_digit(c)) {
        advance = 1;
        return true;
    }
    
    // Кириллица (UTF-8, 2 байта)
    if (pos + 1 < text.size() && is_cyrillic_start(c, text[pos + 1])) {
        advance = 2;
        return true;
    }
    
    // Дефис или апостроф (только внутри слова, проверим позже)
    if (c == '-' || c == '\'') {
        advance = 1;
        return true;
    }
    
    advance = 1;
    return false;
}

std::string Tokenizer::to_lower(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    
    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];
        
        // ASCII латиница A-Z
        if (c >= 'A' && c <= 'Z') {
            result += (char)(c + 32);
            i++;
        }
        // Кириллица UTF-8: А-Я (0xD0 0x90-0xAF) -> а-я (0xD0 0xB0-0xCF)
        else if (i + 1 < str.size() && 
                 (unsigned char)str[i] == 0xD0 && 
                 (unsigned char)str[i+1] >= 0x90 && 
                 (unsigned char)str[i+1] <= 0xAF) {
            result += (char)0xD0;
            result += (char)((unsigned char)str[i+1] + 0x20);
            i += 2;
        }
        // Ё (0xD0 0x81) -> ё (0xD1 0x91)
        else if (i + 1 < str.size() && 
                 (unsigned char)str[i] == 0xD0 && 
                 (unsigned char)str[i+1] == 0x81) {
            result += (char)0xD1;
            result += (char)0x91;
            i += 2;
        }
        else {
            result += str[i];
            i++;
        }
    }
    
    return result;
}

// Подсчет реальной длины в символах (не байтах)
size_t utf8_char_count(const std::string& str) {
    size_t count = 0;
    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];
        
        if (c < 0x80) {
            // ASCII (1 байт)
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            // 2 байта (кириллица)
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3 байта
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4 байта
            i += 4;
        } else {
            i++;
        }
        count++;
    }
    return count;
}

bool Tokenizer::is_valid_token(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    
    // Проверка минимальной длины в символах
    size_t char_count = utf8_char_count(token);
    
    if (char_count < min_token_length_) {
        return false;
    }
    
    // Убираем токены, состоящие только из дефисов/апострофов
    bool has_letter_or_digit = false;
    for (size_t i = 0; i < token.size(); ) {
        unsigned char c = token[i];
        if (is_ascii_letter(c) || is_digit(c)) {
            has_letter_or_digit = true;
            break;
        }
        if (i + 1 < token.size() && is_cyrillic_start(c, token[i + 1])) {
            has_letter_or_digit = true;
            break;
        }
        i++;
    }
    
    return has_letter_or_digit;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    
    size_t i = 0;
    while (i < text.size()) {
        size_t advance = 0;
        
        // Пропускаем не-токен символы
        while (i < text.size() && !is_token_char(text, i, advance)) {
            i++;
        }
        
        // Начало токена
        if (i < text.size()) {
            size_t token_start = i;
            
            // Собираем токен
            while (i < text.size() && is_token_char(text, i, advance)) {
                i += advance;
            }
            
            // Извлекаем токен
            std::string token = text.substr(token_start, i - token_start);
            
            // Убираем дефисы/апострофы в начале и конце
            while (!token.empty() && (token[0] == '-' || token[0] == '\'')) {
                token = token.substr(1);
            }
            while (!token.empty() && (token.back() == '-' || token.back() == '\'')) {
                token.pop_back();
            }
            
            // Приведение к нижнему регистру
            if (lowercase_) {
                token = to_lower(token);
            }
            
            // Проверка валидности
            if (is_valid_token(token)) {
                tokens.push_back(token);
            }
        }
    }
    
    return tokens;
}

std::vector<Tokenizer::Token> Tokenizer::tokenize_with_positions(const std::string& text) {
    std::vector<Token> tokens;
    
    size_t i = 0;
    while (i < text.size()) {
        size_t advance = 0;
        
        // Пропускаем не-токен символы
        while (i < text.size() && !is_token_char(text, i, advance)) {
            i++;
        }
        
        // Начало токена
        if (i < text.size()) {
            size_t token_start = i;
            
            // Собираем токен
            while (i < text.size() && is_token_char(text, i, advance)) {
                i += advance;
            }
            
            // Извлекаем токен
            std::string token_text = text.substr(token_start, i - token_start);
            size_t original_length = token_text.length();
            
            // Убираем дефисы/апострофы в начале и конце
            size_t trim_start = 0;
            while (trim_start < token_text.size() && 
                   (token_text[trim_start] == '-' || token_text[trim_start] == '\'')) {
                trim_start++;
            }
            
            size_t trim_end = 0;
            while (trim_end < token_text.size() - trim_start && 
                   (token_text[token_text.size() - 1 - trim_end] == '-' || 
                    token_text[token_text.size() - 1 - trim_end] == '\'')) {
                trim_end++;
            }
            
            if (trim_start > 0 || trim_end > 0) {
                token_text = token_text.substr(trim_start, token_text.size() - trim_start - trim_end);
            }
            
            // Приведение к нижнему регистру
            if (lowercase_) {
                token_text = to_lower(token_text);
            }
            
            // Проверка валидности
            if (is_valid_token(token_text)) {
                Token token;
                token.text = token_text;
                token.position = token_start + trim_start;
                token.length = original_length - trim_start - trim_end;
                tokens.push_back(token);
            }
        }
    }
    
    return tokens;
}