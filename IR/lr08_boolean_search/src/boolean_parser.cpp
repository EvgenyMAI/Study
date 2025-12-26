#include "boolean_parser.h"
#include <cstring>
#include <cstdlib>
#include <cctype>

// ============ BooleanLexer ============

BooleanLexer::BooleanLexer(const char* query) : input(query), pos(0) {}

void BooleanLexer::reset(const char* query) {
    input = query;
    pos = 0;
}

void BooleanLexer::skip_whitespace() {
    while (input[pos] && isspace(input[pos])) {
        pos++;
    }
}

bool BooleanLexer::is_operator_char(char c) {
    return c == '(' || c == ')' || c == '&' || c == '|' || c == '!';
}

Token BooleanLexer::next_token() {
    skip_whitespace();
    
    Token token;
    token.value[0] = '\0';
    
    if (!input[pos]) {
        token.type = TOKEN_END;
        return token;
    }
    
    // Скобки
    if (input[pos] == '(') {
        token.type = TOKEN_LPAREN;
        strcpy(token.value, "(");
        pos++;
        return token;
    }
    
    if (input[pos] == ')') {
        token.type = TOKEN_RPAREN;
        strcpy(token.value, ")");
        pos++;
        return token;
    }
    
    // Слова (термины и операторы)
    int start = pos;
    while (input[pos] && !isspace(input[pos]) && !is_operator_char(input[pos])) {
        pos++;
    }
    
    int len = pos - start;
    if (len > 255) len = 255;
    strncpy(token.value, input + start, len);
    token.value[len] = '\0';
    
    // Приведение к lowercase для сравнения
    char lower[256];
    for (int i = 0; i <= len; i++) {
        if (token.value[i] >= 'A' && token.value[i] <= 'Z') {
            lower[i] = token.value[i] + 32;
        } else {
            lower[i] = token.value[i];
        }
    }
    
    // Проверка на операторы
    if (strcmp(lower, "and") == 0 || strcmp(lower, "&&") == 0) {
        token.type = TOKEN_AND;
    } else if (strcmp(lower, "or") == 0 || strcmp(lower, "||") == 0) {
        token.type = TOKEN_OR;
    } else if (strcmp(lower, "not") == 0 || strcmp(lower, "!") == 0) {
        token.type = TOKEN_NOT;
    } else {
        token.type = TOKEN_TERM;
        // Для термина сохраняем оригинал в lowercase
        strcpy(token.value, lower);
    }
    
    return token;
}

// ============ BooleanParser ============

BooleanParser::BooleanParser() {
    lexer = new BooleanLexer("");
}

BooleanParser::~BooleanParser() {
    delete lexer;
}

void BooleanParser::advance() {
    current_token = lexer->next_token();
}

void BooleanParser::error(const char* message) {
    fprintf(stderr, "Parse error: %s\n", message);
}

void* BooleanParser::parse(const char* query) {
    lexer->reset(query);
    advance();
    return parse_or();
}

// OR имеет низший приоритет
void* BooleanParser::parse_or() {
    void* left = parse_and();
    
    while (current_token.type == TOKEN_OR) {
        advance();
        void* right = parse_and();
    }
    
    return left;
}

// AND имеет средний приоритет
void* BooleanParser::parse_and() {
    void* left = parse_not();
    
    while (current_token.type == TOKEN_AND || 
           (current_token.type == TOKEN_TERM || current_token.type == TOKEN_LPAREN)) {
        // Неявный AND (два термина подряд)
        if (current_token.type != TOKEN_AND) {
            void* right = parse_not();
            // Создание узла AND
        } else {
            advance();
            void* right = parse_not();
            // Создание узла AND
        }
    }
    
    return left;
}

// NOT имеет высокий приоритет
void* BooleanParser::parse_not() {
    if (current_token.type == TOKEN_NOT) {
        advance();
        void* expr = parse_primary();
        // Создание узла NOT
        return expr;
    }
    
    return parse_primary();
}

// Первичные выражения: термины и скобки
void* BooleanParser::parse_primary() {
    if (current_token.type == TOKEN_TERM) {
        // Создание узла термина
        advance();
        return nullptr;
    }
    
    if (current_token.type == TOKEN_LPAREN) {
        advance();
        void* expr = parse_or();
        if (current_token.type != TOKEN_RPAREN) {
            error("Expected )");
        }
        advance();
        return expr;
    }
    
    error("Unexpected token");
    return nullptr;
}