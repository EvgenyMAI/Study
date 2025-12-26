#ifndef BOOLEAN_QUERY_H
#define BOOLEAN_QUERY_H

#include "boolean_index.h"

class BooleanQuery {
private:
    BooleanIndex* index;

    PostingList* parse_and_execute(const char* query);
    void to_lowercase(char* str);

public:
    BooleanQuery(BooleanIndex* idx);

    PostingList* execute(const char* query);
    void print_results(const PostingList* results);
};

#endif