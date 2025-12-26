#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H

// Используем код из ЛР7
#include "../../lr07_boolean_index/src/posting_list.h"
#include "../../lr07_boolean_index/src/boolean_index.h"

struct SearchResult {
    int doc_id;
    int* term_positions;  // позиции терминов (опционально)
    int num_positions;
};

class SearchEngine {
private:
    BooleanIndex* index;
    
    PostingList* execute_term(const char* term);
    PostingList* execute_and(PostingList* left, PostingList* right);
    PostingList* execute_or(PostingList* left, PostingList* right);
    PostingList* execute_not(PostingList* operand);
    
public:
    SearchEngine(BooleanIndex* idx);
    
    PostingList* search(const char* query);
    void print_results(PostingList* results, int max_results);
};

#endif