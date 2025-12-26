#ifndef SEARCH_H
#define SEARCH_H

#include "stemmer.h"

#define HASH_TABLE_SIZE 500000

struct DocTerm {
    int doc_id;
    int freq;
    DocTerm* next;
};

struct IndexEntry {
    char* term;
    DocTerm* docs;
    IndexEntry* next;
    bool is_finalized;
};

class SearchIndex {
private:
    IndexEntry** hash_table;
    int num_docs;
    int total_terms;
    
    IndexEntry* find_term(const char* term, int hash_value);
    void add_term(const char* term, int doc_id);
    int hash(const char* term);
    void finalize_term(IndexEntry* entry);
    
public:
    SearchIndex();
    ~SearchIndex();
    void add_document(int doc_id, const char* text, RussianStemmer* stemmer, bool use_stemming = true);
    void search(const char* query, RussianStemmer* stemmer, int* results, int* num_results, int max_results, bool use_stemming = true);
    void save(const char* filename);
    void load(const char* filename);
    void print_stats();
    void finalize();
};

#endif