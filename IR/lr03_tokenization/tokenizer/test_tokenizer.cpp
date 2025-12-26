#include "tokenizer.h"
#include <iostream>
#include <cassert>
#include <set>

void test_basic_tokenization() {
    Tokenizer tokenizer;
    
    std::string text = "Hello, world! This is test.";
    auto tokens = tokenizer.tokenize(text);
    
    // Ожидаем: hello, world, this, is, test
    assert(tokens.size() == 5);
    assert(tokens[0] == "hello");
    assert(tokens[1] == "world");
    
    std::cout << "Тест 1: Базовая токенизация" << std::endl;
}

void test_lowercase() {
    Tokenizer tokenizer;
    tokenizer.set_lowercase(true);
    
    std::string text = "Hello WORLD";
    auto tokens = tokenizer.tokenize(text);
    
    assert(tokens.size() == 2);
    assert(tokens[0] == "hello");
    assert(tokens[1] == "world");
    
    std::cout << "Тест 2: Приведение к нижнему регистру" << std::endl;
}

void test_hyphenated_words() {
    Tokenizer tokenizer;
    
    std::string text = "Full-stack developer web-app";
    auto tokens = tokenizer.tokenize(text);
    
    // Дефисные слова должны сохраняться как один токен
    assert(tokens.size() == 3);
    assert(tokens[0] == "full-stack");
    assert(tokens[1] == "developer");
    assert(tokens[2] == "web-app");
    
    std::cout << "Тест 3: Слова через дефис" << std::endl;
}

void test_numbers() {
    Tokenizer tokenizer;
    
    std::string text = "In 2025 there were 123 events";
    auto tokens = tokenizer.tokenize(text);
    
    // Числа должны выделяться как токены
    bool has_2025 = false;
    bool has_123 = false;
    
    for (const auto& token : tokens) {
        if (token == "2025") has_2025 = true;
        if (token == "123") has_123 = true;
    }
    
    assert(has_2025 && has_123);
    
    std::cout << "Тест 4: Числа" << std::endl;
}

void test_min_length() {
    Tokenizer tokenizer;
    tokenizer.set_min_token_length(3);
    
    std::string text = "a in IT go";
    auto tokens = tokenizer.tokenize(text);
    
    // Токены короче 3 символов должны быть пропущены
    // "a" (1), "in" (2), "it" (2), "go" (2) - все < 3
    assert(tokens.size() == 0);
    
    std::cout << "Тест 5: Минимальная длина токена" << std::endl;
}

void test_positions() {
    Tokenizer tokenizer;
    
    std::string text = "one two three";
    auto tokens = tokenizer.tokenize_with_positions(text);
    
    assert(tokens.size() == 3);
    assert(tokens[0].position == 0);   // "one" starts at 0
    assert(tokens[1].position == 4);   // "two" starts at 4
    assert(tokens[2].position == 8);   // "three" starts at 8
    
    std::cout << "Тест 6: Позиции токенов" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Запуск автотестов токенизатора" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        test_basic_tokenization();
        test_lowercase();
        test_hyphenated_words();
        test_numbers();
        test_min_length();
        test_positions();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Все тесты пройдены успешно! (6/6)" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nОшибка: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nНеизвестная ошибка в тестах!" << std::endl;
        return 1;
    }
}