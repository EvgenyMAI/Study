#include "search_engine.h"
#include "boolean_parser.h"
#include <cstdio>
#include <cstring>

SearchEngine::SearchEngine(BooleanIndex* idx) : index(idx) {}

PostingList* SearchEngine::execute_term(const char* term) {
    PostingList* postings = index->get_postings(term);
    if (!postings) {
        return new PostingList(); // Пустой список
    }
    
    // Копируем список
    PostingList* result = new PostingList();
    PostingNode* node = postings->get_head();
    while (node) {
        result->add_document(node->doc_id);
        node = node->next;
    }
    return result;
}

PostingList* SearchEngine::execute_and(PostingList* left, PostingList* right) {
    return PostingList::intersect(left, right);
}

PostingList* SearchEngine::execute_or(PostingList* left, PostingList* right) {
    return PostingList::union_lists(left, right);
}

PostingList* SearchEngine::execute_not(PostingList* operand) {
    // NOT реализуется как разность универсального множества и operand
    PostingList* all_docs = new PostingList();
    for (int i = 0; i < index->get_num_docs(); i++) {
        all_docs->add_document(i);
    }
    
    PostingList* result = PostingList::difference(all_docs, operand);
    delete all_docs;
    return result;
}

PostingList* SearchEngine::search(const char* query) {
    // Простой парсер (без AST, последовательная обработка)
    char query_copy[1000];
    strncpy(query_copy, query, 999);
    query_copy[999] = '\0';
    
    // Токенизация
    char* tokens[100];
    int token_count = 0;
    
    char* token = strtok(query_copy, " \t\n");
    while (token && token_count < 100) {
        tokens[token_count++] = token;
        token = strtok(nullptr, " \t\n");
    }
    
    if (token_count == 0) {
        return new PostingList();
    }
    
    // Приведение к lowercase
    for (int i = 0; i < token_count; i++) {
        for (int j = 0; tokens[i][j]; j++) {
            if (tokens[i][j] >= 'A' && tokens[i][j] <= 'Z') {
                tokens[i][j] = tokens[i][j] + 32;
            }
        }
    }
    
    // Выполнение запроса
    PostingList* result = nullptr;
    char current_op[10] = "AND";
    bool negate_next = false;
    
    for (int i = 0; i < token_count; i++) {
        char* term = tokens[i];
        
        if (strcmp(term, "and") == 0 || strcmp(term, "&&") == 0) {
            strcpy(current_op, "AND");
            continue;
        } else if (strcmp(term, "or") == 0 || strcmp(term, "||") == 0) {
            strcpy(current_op, "OR");
            continue;
        } else if (strcmp(term, "not") == 0 || strcmp(term, "!") == 0) {
            negate_next = true;
            continue;
        }
        
        // Игнорируем скобки в простой версии
        if (strcmp(term, "(") == 0 || strcmp(term, ")") == 0) {
            continue;
        }
        
        // Получаем posting list для термина
        PostingList* term_list = execute_term(term);
        
        // Применяем NOT если нужно
        if (negate_next) {
            PostingList* negated = execute_not(term_list);
            delete term_list;
            term_list = negated;
            negate_next = false;
        }
        
        // Применяем операцию
        if (!result) {
            result = term_list;
        } else {
            if (strcmp(current_op, "AND") == 0) {
                PostingList* temp = execute_and(result, term_list);
                delete result;
                delete term_list;
                result = temp;
            } else if (strcmp(current_op, "OR") == 0) {
                PostingList* temp = execute_or(result, term_list);
                delete result;
                delete term_list;
                result = temp;
            }
            strcpy(current_op, "AND");
        }
    }
    
    return result ? result : new PostingList();
}

void SearchEngine::print_results(PostingList* results, int max_results) {
    if (!results || results->get_size() == 0) {
        printf("0\n");
        return;
    }
    
    printf("%d\n", results->get_size());
    
    PostingNode* node = results->get_head();
    int count = 0;
    while (node && (max_results == -1 || count < max_results)) {
        printf("%d\n", node->doc_id);
        node = node->next;
        count++;
    }
}