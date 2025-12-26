#include "search.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int compare_doc_freq(const void* a, const void* b) {
    struct DocFreq {
        int doc_id;
        int freq;
    };
    const DocFreq* da = (const DocFreq*)a;
    const DocFreq* db = (const DocFreq*)b;
    return da->doc_id - db->doc_id;
}

static int compare_doc_score(const void* a, const void* b) {
    struct DocScore {
        int doc_id;
        int score;
    };
    const DocScore* da = (const DocScore*)a;
    const DocScore* db = (const DocScore*)b;
    return db->score - da->score;  // По убыванию
}

SearchIndex::SearchIndex() : num_docs(0), total_terms(0) {
    hash_table = (IndexEntry**)calloc(HASH_TABLE_SIZE, sizeof(IndexEntry*));
}

SearchIndex::~SearchIndex() {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* curr = hash_table[i];
        while (curr) {
            IndexEntry* next = curr->next;
            
            DocTerm* doc = curr->docs;
            while (doc) {
                DocTerm* next_doc = doc->next;
                free(doc);
                doc = next_doc;
            }
            
            free(curr->term);
            free(curr);
            curr = next;
        }
    }
    free(hash_table);
}

int SearchIndex::hash(const char* term) {
    unsigned long hash = 5381;
    int c;
    while ((c = *term++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_TABLE_SIZE;
}

IndexEntry* SearchIndex::find_term(const char* term, int hash_value) {
    IndexEntry* curr = hash_table[hash_value];
    while (curr) {
        if (strcmp(curr->term, term) == 0) {
            return curr;
        }
        curr = curr->next;
    }
    return nullptr;
}

void SearchIndex::add_term(const char* term, int doc_id) {
    int hash_value = hash(term);
    IndexEntry* entry = find_term(term, hash_value);
    
    if (!entry) {
        entry = (IndexEntry*)malloc(sizeof(IndexEntry));
        entry->term = (char*)malloc(strlen(term) + 1);
        strcpy(entry->term, term);
        entry->docs = nullptr;
        entry->next = hash_table[hash_value];
        entry->is_finalized = false;
        hash_table[hash_value] = entry;
        total_terms++;
    }
    
    // Дедупликация будет после индексации
    DocTerm* new_doc = (DocTerm*)malloc(sizeof(DocTerm));
    new_doc->doc_id = doc_id;
    new_doc->freq = 1;
    new_doc->next = entry->docs;
    entry->docs = new_doc;
    
}

void SearchIndex::finalize() {
    printf("\nFinalizing index (deduplication and sorting)...\n");
    
    int processed = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* entry = hash_table[i];
        while (entry) {
            if (!entry->is_finalized) {
                finalize_term(entry);
                entry->is_finalized = true;
                
                processed++;
                if (processed % 10000 == 0) {
                    printf("Finalized %d terms...\r", processed);
                    fflush(stdout);
                }
            }
            entry = entry->next;
        }
    }
    printf("\nFinalization complete: %d terms\n", processed);
}

void SearchIndex::finalize_term(IndexEntry* entry) {
    if (!entry->docs) return;
    
    // 1. Подсчитываем размер
    int count = 0;
    DocTerm* doc = entry->docs;
    while (doc) {
        count++;
        doc = doc->next;
    }
    
    if (count == 0) return;
    
    // 2. Копируем в массив
    struct DocFreq {
        int doc_id;
        int freq;
    };
    DocFreq* arr = (DocFreq*)malloc(count * sizeof(DocFreq));
    
    doc = entry->docs;
    int idx = 0;
    while (doc) {
        arr[idx].doc_id = doc->doc_id;
        arr[idx].freq = doc->freq;
        idx++;
        doc = doc->next;
    }
    
    // 3. Сортируем по doc_id через обычную функцию
    qsort(arr, count, sizeof(DocFreq), compare_doc_freq);
    
    // 4. Дедупликация и суммирование частот
    int write_pos = 0;
    for (int read_pos = 0; read_pos < count; read_pos++) {
        if (write_pos == 0 || arr[write_pos-1].doc_id != arr[read_pos].doc_id) {
            arr[write_pos] = arr[read_pos];
            write_pos++;
        } else {
            arr[write_pos-1].freq += arr[read_pos].freq;
        }
    }
    
    int final_count = write_pos;
    
    // 5. Освобождаем старый список
    doc = entry->docs;
    while (doc) {
        DocTerm* next = doc->next;
        free(doc);
        doc = next;
    }
    
    // 6. Создаем новый отсортированный список
    entry->docs = nullptr;
    for (int i = final_count - 1; i >= 0; i--) {
        DocTerm* new_doc = (DocTerm*)malloc(sizeof(DocTerm));
        new_doc->doc_id = arr[i].doc_id;
        new_doc->freq = arr[i].freq;
        new_doc->next = entry->docs;
        entry->docs = new_doc;
    }
    
    free(arr);
}

void SearchIndex::add_document(int doc_id, const char* text, 
                                RussianStemmer* stemmer, bool use_stemming) {
    char* text_copy = (char*)malloc(strlen(text) + 1);
    strcpy(text_copy, text);
    
    char* token = strtok(text_copy, " \t\n\r.,;:!?()[]{}\"'<>/\\|@#$%^&*+=`~");
    int token_count = 0;
    
    while (token) {
        if (strlen(token) > 2) {
            // Lowercase
            for (int i = 0; token[i]; i++) {
                if (token[i] >= 'A' && token[i] <= 'Z') {
                    token[i] = token[i] + 32;
                }
            }
            
            const char* term;
            if (use_stemming) {
                term = stemmer->stem(token);
            } else {
                term = token;
            }
            
            if (strlen(term) > 2) {
                add_term(term, doc_id);
                token_count++;
            }
        }
        token = strtok(nullptr, " \t\n\r.,;:!?()[]{}\"'<>/\\|@#$%^&*+=`~");
    }
    
    free(text_copy);
    
    if (doc_id >= num_docs) {
        num_docs = doc_id + 1;
    }
}

void SearchIndex::search(const char* query, RussianStemmer* stemmer, 
                        int* results, int* num_results, int max_results, 
                        bool use_stemming) {
    *num_results = 0;
    
    int* scores = (int*)calloc(num_docs, sizeof(int));
    
    char* query_copy = (char*)malloc(strlen(query) + 1);
    strcpy(query_copy, query);
    
    char* token = strtok(query_copy, " \t\n\r.,;:!?()[]{}\"'");
    while (token) {
        if (strlen(token) > 2) {
            for (int i = 0; token[i]; i++) {
                if (token[i] >= 'A' && token[i] <= 'Z') {
                    token[i] = token[i] + 32;
                }
            }
            
            const char* term;
            if (use_stemming) {
                term = stemmer->stem(token);
            } else {
                term = token;
            }
            
            int hash_value = hash(term);
            IndexEntry* entry = find_term(term, hash_value);
            
            if (entry) {
                DocTerm* doc = entry->docs;
                while (doc) {
                    scores[doc->doc_id] += doc->freq;
                    doc = doc->next;
                }
            }
        }
        token = strtok(nullptr, " \t\n\r.,;:!?()[]{}\"'");
    }
    
    free(query_copy);
    
    struct DocScore {
        int doc_id;
        int score;
    };
    
    int non_zero_count = 0;
    for (int j = 0; j < num_docs; j++) {
        if (scores[j] > 0) non_zero_count++;
    }
    
    if (non_zero_count == 0) {
        free(scores);
        return;
    }
    
    DocScore* doc_scores = (DocScore*)malloc(non_zero_count * sizeof(DocScore));
    int count = 0;
    
    for (int j = 0; j < num_docs; j++) {
        if (scores[j] > 0) {
            doc_scores[count].doc_id = j;
            doc_scores[count].score = scores[j];
            count++;
        }
    }
    
    qsort(doc_scores, count, sizeof(DocScore), compare_doc_score);
    
    *num_results = (count < max_results) ? count : max_results;
    for (int i = 0; i < *num_results; i++) {
        results[i] = doc_scores[i].doc_id;
    }
    
    free(doc_scores);
    free(scores);
}

void SearchIndex::print_stats() {
    printf("Index statistics:\n");
    printf("Documents: %d\n", num_docs);
    printf("Unique terms: %d\n", total_terms);
    
    // Подсчитываем коллизии
    int used_buckets = 0;
    int max_chain = 0;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (hash_table[i]) {
            used_buckets++;
            int chain_len = 0;
            IndexEntry* curr = hash_table[i];
            while (curr) {
                chain_len++;
                curr = curr->next;
            }
            if (chain_len > max_chain) {
                max_chain = chain_len;
            }
        }
    }
    
    printf("Hash table load: %.2f%%\n", (used_buckets * 100.0) / HASH_TABLE_SIZE);
    printf("Max chain length: %d\n", max_chain);
}

void SearchIndex::save(const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open file for writing: %s\n", filename);
        return;
    }
    
    fprintf(f, "%d\n", num_docs);
    fprintf(f, "%d\n", total_terms);
    
    int saved = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        IndexEntry* curr = hash_table[i];
        while (curr) {
            fprintf(f, "%s\n", curr->term);
            
            // Подсчитываем документы в финализированном списке
            int doc_count = 0;
            DocTerm* doc = curr->docs;
            while (doc) {
                doc_count++;
                doc = doc->next;
            }
            fprintf(f, "%d\n", doc_count);
            
            // Сохраняем с частотами
            doc = curr->docs;
            while (doc) {
                fprintf(f, "%d %d\n", doc->doc_id, doc->freq);
                doc = doc->next;
            }
            
            saved++;
            if (saved % 10000 == 0) {
                fprintf(stderr, "Saved %d/%d terms...\r", saved, total_terms);
                fflush(stderr);
            }
            
            curr = curr->next;
        }
    }
    
    fclose(f);
    fprintf(stderr, "\nIndex saved: %d terms\n", saved);
}

void SearchIndex::load(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open file for reading: %s\n", filename);
        return;
    }
    
    fscanf(f, "%d\n", &num_docs);
    fscanf(f, "%d\n", &total_terms);
    
    fprintf(stderr, "Loading index: %d documents, %d terms\n", num_docs, total_terms);
    
    char term[256];
    int loaded = 0;
    
    while (fscanf(f, "%255s\n", term) == 1) {
        int doc_count;
        fscanf(f, "%d\n", &doc_count);
        
        // Не вызываем add_term, а напрямую добавляем
        int hash_value = hash(term);
        
        // Создаем entry если его нет
        IndexEntry* entry = find_term(term, hash_value);
        if (!entry) {
            entry = (IndexEntry*)malloc(sizeof(IndexEntry));
            entry->term = (char*)malloc(strlen(term) + 1);
            strcpy(entry->term, term);
            entry->docs = nullptr;
            entry->next = hash_table[hash_value];
            entry->is_finalized = true;  // Уже финализирован
            hash_table[hash_value] = entry;
        }
        
        // Загружаем документы с частотами
        for (int i = 0; i < doc_count; i++) {
            int doc_id, freq;
            fscanf(f, "%d %d\n", &doc_id, &freq);
            
            DocTerm* new_doc = (DocTerm*)malloc(sizeof(DocTerm));
            new_doc->doc_id = doc_id;
            new_doc->freq = freq;
            new_doc->next = entry->docs;
            entry->docs = new_doc;
        }
        
        loaded++;
        if (loaded % 10000 == 0) {
            fprintf(stderr, "Loaded %d/%d terms...\r", loaded, total_terms);
            fflush(stderr);
        }
    }
    
    fclose(f);
    fprintf(stderr, "\nIndex loaded: %d terms\n", loaded);
}