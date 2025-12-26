#ifndef BOOLEAN_INDEX_H
#define BOOLEAN_INDEX_H

#include "posting_list.h"

#define HASH_TABLE_SIZE 200000

struct IndexEntry {
    char* term;
    PostingList* postings;
    IndexEntry* next;
};

class BooleanIndex {
private:
    IndexEntry** hash_table;
    int num_terms;
    int num_docs;

    int hash(const char* term);
    IndexEntry* find_term(const char* term);

public:
    BooleanIndex();
    ~BooleanIndex();

    void add_term(const char* term, int doc_id);
    PostingList* get_postings(const char* term);

    void set_num_docs(int n);
    int get_num_docs() const;
    int get_num_terms() const;

    // Финализация индекса (сортировка всех posting lists)
    void finalize();

    void save(const char* filename);
    void load(const char* filename);

    void print_stats();
};

#endif