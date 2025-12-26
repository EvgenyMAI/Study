#include "boolean_query.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

BooleanQuery::BooleanQuery(BooleanIndex* idx) : index(idx) {}

void BooleanQuery::to_lowercase(char* str) {
    for (int i = 0; str[i]; i++) {
        if (str[i] >= 'A' && str[i] <= 'Z') {
            str[i] = str[i] + 32;
        }
    }
}

PostingList* BooleanQuery::execute(const char* query) {
    return parse_and_execute(query);
}

PostingList* BooleanQuery::parse_and_execute(const char* query) {
    char* query_copy = (char*)malloc(strlen(query) + 1);
    strcpy(query_copy, query);

    // Токенизация
    char* tokens[100];
    int token_count = 0;

    char* token = strtok(query_copy, " \t\n");
    while (token && token_count < 100) {
        tokens[token_count++] = token;
        token = strtok(nullptr, " \t\n");
    }

    if (token_count == 0) {
        free(query_copy);
        return new PostingList();
    }

    PostingList* result = nullptr;
    char current_op[10] = "AND";
    bool negate_next = false;

    for (int i = 0; i < token_count; i++) {
        char* term = tokens[i];
        to_lowercase(term);

        if (strcmp(term, "and") == 0) {
            strcpy(current_op, "AND");
            continue;
        } else if (strcmp(term, "or") == 0) {
            strcpy(current_op, "OR");
            continue;
        } else if (strcmp(term, "not") == 0) {
            negate_next = true;
            continue;
        }

        // Получаем posting list для термина
        PostingList* original_list = index->get_postings(term);
        PostingList* term_list = nullptr;

        if (!original_list || original_list->get_size() == 0) {
            term_list = new PostingList();
        } else {
            // Применяем NOT если нужно
            if (negate_next) {
                // Создаем список всех документов
                PostingList* all_docs = new PostingList();
                for (int j = 0; j < index->get_num_docs(); j++) {
                    all_docs->add_document(j);
                }
                all_docs->finalize();

                // Вычитаем term_list
                term_list = PostingList::difference(all_docs, original_list);
                delete all_docs;
                negate_next = false;
            } else {
                // Быстрое копирование через memcpy
                term_list = original_list->copy();
            }
        }

        // Применяем операцию
        if (!result) {
            result = term_list;
        } else {
            PostingList* temp = nullptr;

            if (strcmp(current_op, "AND") == 0) {
                temp = PostingList::intersect(result, term_list);
            } else if (strcmp(current_op, "OR") == 0) {
                temp = PostingList::union_lists(result, term_list);
            }

            delete result;
            delete term_list;
            result = temp;
            strcpy(current_op, "AND");
        }
    }

    free(query_copy);
    return result ? result : new PostingList();
}

void BooleanQuery::print_results(const PostingList* results) {
    if (!results || results->get_size() == 0) {
        printf("No documents found.\n");
        return;
    }

    printf("Found %d documents:\n", results->get_size());
    int count = 0;
    for (int i = 0; i < results->get_size() && count < 20; i++, count++) {
        printf("Document ID: %d\n", results->get_document(i));
    }
    if (results->get_size() > count) {
        printf("... and %d more\n", results->get_size() - count);
    }
}