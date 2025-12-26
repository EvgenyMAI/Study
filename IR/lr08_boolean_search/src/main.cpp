#include <cstdio>
#include <cstring>
#include "../../lr07_boolean_index/src/boolean_index.h"
#include "search_engine.h"

void print_usage() {
    printf("Usage:\n");
    printf("  boolean_search <index_file> <query>  - Search with boolean query\n");
    printf("  boolean_search <index_file>          - Interactive mode (read from stdin)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    // Загрузка индекса
    BooleanIndex index;
    fprintf(stderr, "Loading index from %s...\n", argv[1]);
    index.load(argv[1]);
    fprintf(stderr, "Index loaded: %d documents, %d terms\n", 
            index.get_num_docs(), index.get_num_terms());
    
    SearchEngine engine(&index);
    
    if (argc >= 3) {
        // Запрос из аргументов командной строки
        char query[1000] = "";
        for (int i = 2; i < argc; i++) {
            strcat(query, argv[i]);
            if (i < argc - 1) strcat(query, " ");
        }
        
        PostingList* results = engine.search(query);
        engine.print_results(results, 1000);
        delete results;
    } else {
        // Интерактивный режим - чтение из stdin
        fprintf(stderr, "Enter queries (one per line), Ctrl+D to exit:\n");
        char query[1000];
        while (fgets(query, sizeof(query), stdin)) {
            // Удаляем \n
            int len = strlen(query);
            if (len > 0 && query[len-1] == '\n') {
                query[len-1] = '\0';
            }
            
            if (strlen(query) == 0) continue;
            
            PostingList* results = engine.search(query);
            engine.print_results(results, 1000);
            delete results;
        }
    }
    
    return 0;
}