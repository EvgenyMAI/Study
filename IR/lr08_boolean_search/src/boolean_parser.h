#ifndef BOOLEAN_PARSER_H
#define BOOLEAN_PARSER_H

#include <cstdio>

// Типы токенов
enum TokenType {
    TOKEN_TERM,      // слово
    TOKEN_AND,       // AND
    TOKEN_OR,        // OR
    TOKEN_NOT,       // NOT
    TOKEN_LPAREN,    // (
    TOKEN_RPAREN,    // )
    TOKEN_END        // конец
};

struct Token {
    TokenType type;
    char value[256];
};

// Лексер для разбора запроса
class BooleanLexer {
private:
    const char* input;
    int pos;
    
    void skip_whitespace();
    bool is_operator_char(char c);
    
public:
    BooleanLexer(const char* query);
    Token next_token();
    void reset(const char* query);
};

// Парсер булевых выражений с приоритетами
class BooleanParser {
private:
    BooleanLexer* lexer;
    Token current_token;
    
    void advance();
    void error(const char* message);
    
    // Рекурсивный спуск с приоритетами: OR < AND < NOT < TERM
    void* parse_or();
    void* parse_and();
    void* parse_not();
    void* parse_primary();
    
public:
    BooleanParser();
    ~BooleanParser();
    
    void* parse(const char* query);
};

#endif