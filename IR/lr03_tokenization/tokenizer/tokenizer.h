#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <regex>
#include <algorithm>

class Tokenizer {
public:
    Tokenizer();
    
    // Токенизация одного текста
    std::vector<std::string> tokenize(const std::string& text);
    
    // Токенизация с сохранением позиций
    struct Token {
        std::string text;
        size_t position;
        size_t length;
    };
    
    std::vector<Token> tokenize_with_positions(const std::string& text);
    
    // Настройки
    void set_lowercase(bool value) { lowercase_ = value; }
    void set_remove_punctuation(bool value) { remove_punctuation_ = value; }
    void set_min_token_length(size_t value) { min_token_length_ = value; }
    
private:
    bool lowercase_;
    bool remove_punctuation_;
    size_t min_token_length_;
    
    // Оставлены для совместимости (не используются)
    std::regex word_pattern_;
    std::regex punctuation_pattern_;
    
    // Вспомогательные методы
    std::string to_lower(const std::string& str);
    bool is_valid_token(const std::string& token);
};

#endif // TOKENIZER_H