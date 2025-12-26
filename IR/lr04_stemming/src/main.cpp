#include <iostream>
#include <cstring>
#include "stemmer.h"
#include "search.h"

void print_usage() {
    printf("Usage:\n");
    printf("stemmer stem <word>           - Stem a single word\n");
    printf("stemmer index <input> <out>   - Build search index\n");
    printf("stemmer search <index> <query>- Search in index\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    RussianStemmer stemmer;
    
    if (strcmp(argv[1], "stem") == 0 && argc == 3) {
        char* result = stemmer.stem(argv[2]);
        printf("%s\n", result);
    }
    else if (strcmp(argv[1], "index") == 0 && argc == 4) {
        SearchIndex index;
        
        FILE* f = fopen(argv[2], "r");
        if (!f) {
            fprintf(stderr, "Cannot open input file: %s\n", argv[2]);
            return 1;
        }
        
        // Проверяем размер файла
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        printf("Building index from %s (%.2f MB)...\n", 
            argv[2], file_size / (1024.0 * 1024.0));
        
        // Безопасное выделение памяти с проверкой
        char* file_content = (char*)malloc(file_size + 1);
        if (!file_content) {
            fprintf(stderr, "ERROR: Cannot allocate %.2f MB for file!\n", 
                    file_size / (1024.0 * 1024.0));
            fclose(f);
            return 1;
        }
        
        // Читаем файл
        size_t bytes_read = fread(file_content, 1, file_size, f);
        file_content[bytes_read] = '\0';
        fclose(f);
        
        printf("File loaded: %zu bytes\n", bytes_read);
        
        // Разбиваем по разделителю
        const char* delimiter = "\n---DOCUMENT---\n";
        int delimiter_len = strlen(delimiter);
        
        char* doc_start = file_content;
        char* doc_end;
        int doc_id = 0;
        
        while ((doc_end = strstr(doc_start, delimiter)) != NULL) {
            *doc_end = '\0';  // Обрезаем
            
            int doc_len = strlen(doc_start);
            if (doc_len > 10) {
                index.add_document(doc_id, doc_start, &stemmer);
                doc_id++;
                
                if (doc_id % 100 == 0) {
                    printf("Indexed %d documents...\r", doc_id);
                    fflush(stdout);
                }
            }
            
            doc_start = doc_end + delimiter_len;
        }
        
        // Последний документ
        if (strlen(doc_start) > 10) {
            index.add_document(doc_id, doc_start, &stemmer);
            doc_id++;
        }
        
        free(file_content);
        
        printf("\n\nIndexed %d documents\n", doc_id);
        
        // Финализация
        index.finalize();
        
        printf("\nIndexing statistics:\n");
        index.print_stats();
        
        printf("\nSaving index to %s...\n", argv[3]);
        index.save(argv[3]);
        
        printf("Index build complete!\n");
    }
    else if (strcmp(argv[1], "search") == 0 && argc == 4) {
        SearchIndex index;
        // Логи в stderr вместо stdout
        fprintf(stderr, "Loading index from %s...\n", argv[2]);
        index.load(argv[2]);
        
        int results[100];
        int num_results;
        index.search(argv[3], &stemmer, results, &num_results, 100);
        
        // Только результаты в stdout
        printf("%d\n", num_results);
        for (int i = 0; i < num_results; i++) {
            printf("%d\n", results[i]);
        }
    }
    else {
        print_usage();
        return 1;
    }
    
    return 0;
}