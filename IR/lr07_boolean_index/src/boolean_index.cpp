#include "boolean_index.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

BooleanIndex::BooleanIndex() : num_terms(0), num_docs(0) {
    hash_table = (IndexEntry**)calloc(HASH_TABLE_SIZE, sizeof(IndexEntry*));
}

BooleanIndex::~BooleanIndex() {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* current = hash_table[i];
        while (current) {
            IndexEntry* next = current->next;
            free(current->term);
            delete current->postings;
            free(current);
            current = next;
        }
    }
    free(hash_table);
}

int BooleanIndex::hash(const char* term) {
    unsigned long hash = 5381;
    int c;
    while ((c = *term++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_TABLE_SIZE;
}

IndexEntry* BooleanIndex::find_term(const char* term) {
    int hash_value = hash(term);
    IndexEntry* current = hash_table[hash_value];

    while (current) {
        if (strcmp(current->term, term) == 0) {
            return current;
        }
        current = current->next;
    }

    return nullptr;
}

void BooleanIndex::add_term(const char* term, int doc_id) {
    IndexEntry* entry = find_term(term);

    if (!entry) {
        int hash_value = hash(term);
        entry = (IndexEntry*)malloc(sizeof(IndexEntry));
        entry->term = (char*)malloc(strlen(term) + 1);
        strcpy(entry->term, term);
        entry->postings = new PostingList();
        entry->next = hash_table[hash_value];
        hash_table[hash_value] = entry;
        num_terms++;
    }

    entry->postings->add_document(doc_id);
}

PostingList* BooleanIndex::get_postings(const char* term) {
    IndexEntry* entry = find_term(term);
    return entry ? entry->postings : nullptr;
}

void BooleanIndex::set_num_docs(int n) {
    num_docs = n;
}

int BooleanIndex::get_num_docs() const {
    return num_docs;
}

int BooleanIndex::get_num_terms() const {
    return num_terms;
}

void BooleanIndex::finalize() {
    fprintf(stderr, "Finalizing index (sorting posting lists)...\n");

    int processed = 0;
    int last_reported = 0;

    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* current = hash_table[i];
        while (current) {
            current->postings->finalize();
            current = current->next;
            processed++;

            // Выводим прогресс каждые 100000 терминов
            if (processed - last_reported >= 100000) {
                fprintf(stderr, "Finalized %d/%d terms...\n", processed, num_terms);
                last_reported = processed;
            }
        }
    }

    fprintf(stderr, "Finalization complete!\n");
}

void BooleanIndex::save(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open file for writing: %s\n", filename);
        return;
    }

    fprintf(f, "%d %d\n", num_docs, num_terms);

    int saved = 0;
    int last_reported = 0;

    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* current = hash_table[i];
        while (current) {
            fprintf(f, "%s\n", current->term);
            current->postings->save(f);

            saved++;
            // Выводим прогресс каждые 100000 терминов
            if (saved - last_reported >= 100000) {
                fprintf(stderr, "Saved %d/%d terms...\n", saved, num_terms);
                last_reported = saved;
            }

            current = current->next;
        }
    }

    fclose(f);
    fprintf(stderr, "Index saved: %d terms\n", saved);
}

void BooleanIndex::load(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open file for reading: %s\n", filename);
        return;
    }

    fscanf(f, "%d %d\n", &num_docs, &num_terms);
    fprintf(stderr, "Loading index: %d documents, %d terms\n", num_docs, num_terms);

    char term[256];
    int loaded = 0;
    int last_reported = 0;

    while (fscanf(f, "%255s\n", term) == 1) {
        int hash_value = hash(term);

        IndexEntry* entry = (IndexEntry*)malloc(sizeof(IndexEntry));
        entry->term = (char*)malloc(strlen(term) + 1);
        strcpy(entry->term, term);
        entry->postings = new PostingList();
        entry->postings->load(f);
        entry->next = hash_table[hash_value];
        hash_table[hash_value] = entry;

        loaded++;
        // Выводим прогресс каждые 500000 терминов
        if (loaded - last_reported >= 500000) {
            fprintf(stderr, "Loaded %d/%d terms (%.1f%%)...\n", 
                    loaded, num_terms, (loaded * 100.0) / num_terms);
            last_reported = loaded;
        }
    }

    fclose(f);
    fprintf(stderr, "Index loaded: %d terms\n", loaded);
}

void BooleanIndex::print_stats() {
    printf("Boolean Index Statistics:\n");
    printf("Documents: %d\n", num_docs);
    printf("Terms: %d\n", num_terms);

    int used_buckets = 0;
    int max_chain = 0;
    long long total_postings = 0;

    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (hash_table[i]) {
            used_buckets++;
            int chain_len = 0;
            IndexEntry* current = hash_table[i];
            while (current) {
                chain_len++;
                total_postings += current->postings->get_size();
                current = current->next;
            }
            if (chain_len > max_chain) {
                max_chain = chain_len;
            }
        }
    }

    printf("Total postings: %lld\n", total_postings);
    printf("Avg postings per term: %.2f\n", (double)total_postings / num_terms);
    printf("Hash table load: %.2f%%\n", (used_buckets * 100.0) / HASH_TABLE_SIZE);
    printf("Max chain length: %d\n", max_chain);
}