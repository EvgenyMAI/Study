#include <iostream>
#include <cstring>
#include "stemmer.h"
#include "search.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: interactive <index_file>\n");
        return 1;
    }
    
    SearchIndex index;
    RussianStemmer stemmer;
    
    fprintf(stderr, "Loading index from %s...\n", argv[1]);
    index.load(argv[1]);
    fprintf(stderr, "Index loaded. Enter queries (one per line), Ctrl+D to exit:\n");
    
    char query[1000];
    while (fgets(query, sizeof(query), stdin)) {
        // Удаляем перевод строки
        query[strcspn(query, "\n")] = 0;
        
        if (strlen(query) == 0) continue;
        
        int results[100];
        int num_results;
        index.search(query, &stemmer, results, &num_results, 100);
        
        printf("Found: %d documents\n", num_results);
        for (int i = 0; i < num_results && i < 10; i++) {
            printf("  %d. Document ID: %d\n", i+1, results[i]);
        }
        printf("\n");
    }
    
    return 0;
}