#include <cstdio>
#include <cstring>
#include <ctime>
#include "boolean_index.h"
#include "boolean_query.h"

void print_usage() {
    printf("Usage:\n");
    printf("boolean_index build <input> <output>  - Build boolean index\n");
    printf("boolean_index search <index> <query>  - Search using boolean query\n");
    printf("boolean_index stats <index>           - Show index statistics\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    if (strcmp(argv[1], "build") == 0 && argc == 4) {
        BooleanIndex index;

        FILE* f = fopen(argv[2], "r");
        if (!f) {
            fprintf(stderr, "Cannot open input file: %s\n", argv[2]);
            return 1;
        }

        printf("Building boolean index from %s...\n", argv[2]);
        clock_t start_time = clock();

        char buffer[500000];
        int doc_id = 0;

        while (fgets(buffer, sizeof(buffer), f)) {
            if (strlen(buffer) > 10) {
                // Токенизация
                char* token = strtok(buffer, " \t\n\r.,;:!?()[]{}\"'<>/\\|@#$%^&*+=`~");
                while (token) {
                    if (strlen(token) > 2) {
                        // Приведение к lowercase
                        for (int i = 0; token[i]; i++) {
                            if (token[i] >= 'A' && token[i] <= 'Z') {
                                token[i] = token[i] + 32;
                            }
                        }
                        index.add_term(token, doc_id);
                    }
                    token = strtok(nullptr, " \t\n\r.,;:!?()[]{}\"'<>/\\|@#$%^&*+=`~");
                }
                doc_id++;

                if (doc_id % 1000 == 0) {
                    printf("Indexed %d documents...\r", doc_id);
                    fflush(stdout);
                }
            }
        }
        fclose(f);

        index.set_num_docs(doc_id);

        clock_t index_time = clock();
        double index_seconds = (double)(index_time - start_time) / CLOCKS_PER_SEC;

        printf("\n\nIndexing complete in %.2f seconds!\n", index_seconds);

        // Финализация индекса (сортировка posting lists)
        printf("\nFinalizing index...\n");
        clock_t finalize_start = clock();
        index.finalize();
        clock_t finalize_end = clock();
        double finalize_seconds = (double)(finalize_end - finalize_start) / CLOCKS_PER_SEC;
        printf("Finalization completed in %.2f seconds!\n", finalize_seconds);

        index.print_stats();

        printf("\nSaving index to %s...\n", argv[3]);
        clock_t save_start = clock();
        index.save(argv[3]);
        clock_t save_end = clock();
        double save_seconds = (double)(save_end - save_start) / CLOCKS_PER_SEC;

        printf("\nDone! Total time: %.2f seconds\n", 
               index_seconds + finalize_seconds + save_seconds);
        printf("  - Indexing: %.2f s\n", index_seconds);
        printf("  - Finalization: %.2f s\n", finalize_seconds);
        printf("  - Saving: %.2f s\n", save_seconds);
        printf("Indexed %d documents\n", doc_id);
    }
    else if (strcmp(argv[1], "search") == 0 && argc >= 4) {
        BooleanIndex index;

        fprintf(stderr, "Loading index from %s...\n", argv[2]);
        clock_t load_start = clock();
        index.load(argv[2]);
        clock_t load_end = clock();
        double load_time = (double)(load_end - load_start) / CLOCKS_PER_SEC;
        fprintf(stderr, "Index loaded in %.2f seconds\n", load_time);

        // Собираем запрос из всех аргументов
        char query[1000] = "";
        for (int i = 3; i < argc; i++) {
            strcat(query, argv[i]);
            if (i < argc - 1) strcat(query, " ");
        }

        fprintf(stderr, "Executing query: %s\n", query);
        clock_t search_start = clock();

        BooleanQuery bq(&index);
        PostingList* results = bq.execute(query);

        clock_t search_end = clock();
        double search_time = (double)(search_end - search_start) / CLOCKS_PER_SEC;

        fprintf(stderr, "Search completed in %.3f seconds\n", search_time);
        fprintf(stderr, "Found %d documents\n", results->get_size());

        // выводим только первые 10000 результатов
        const int MAX_RESULTS = 10000;
        int output_count = results->get_size() < MAX_RESULTS ? results->get_size() : MAX_RESULTS;

        // Вывод в формате для парсинга Python
        printf("%d\n", results->get_size());  // Общее количество
        for (int i = 0; i < output_count; i++) {
            printf("%d\n", results->get_document(i));
        }

        if (results->get_size() > MAX_RESULTS) {
            fprintf(stderr, "Note: Only first %d results printed (total: %d)\n", 
                    MAX_RESULTS, results->get_size());
        }

        delete results;
    }
    else if (strcmp(argv[1], "stats") == 0 && argc == 3) {
        BooleanIndex index;

        printf("Loading index from %s...\n", argv[2]);
        index.load(argv[2]);

        printf("\n");
        index.print_stats();
    }
    else {
        print_usage();
        return 1;
    }

    return 0;
}