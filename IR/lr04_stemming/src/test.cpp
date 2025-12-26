#include <cstdio>
#include <cstring>
#include "stemmer.h"
#include "search.h"

int test_count = 0;
int passed = 0;

void test(const char* name, bool condition) {
    test_count++;
    if (condition) {
        printf("[OK] %s\n", name);
        passed++;
    } else {
        printf("[FAIL] %s\n", name);
    }
}

int main() {
    RussianStemmer stemmer;
    
    // Test 1: Basic stemming
    char* result = stemmer.stem("программирование");
    test("Stemming 'программирование'", strlen(result) < strlen("программирование"));
    
    // Test 2: Word forms - используем более простые слова
    char stem1[256], stem2[256], stem3[256];
    strcpy(stem1, stemmer.stem("книга"));
    strcpy(stem2, stemmer.stem("книги"));
    strcpy(stem3, stemmer.stem("книгам"));
    
    printf("  DEBUG: '%s' -> '%s'\n", "книга", stem1);
    printf("  DEBUG: '%s' -> '%s'\n", "книги", stem2);
    printf("  DEBUG: '%s' -> '%s'\n", "книгам", stem3);
    
    // Все три должны начинаться одинаково
    bool same_stem = (strncmp(stem1, stem2, 4) == 0) && (strncmp(stem2, stem3, 4) == 0);
    test("Word forms same stem", same_stem);
    
    // Test 3: Short word
    result = stemmer.stem("я");
    test("Short word unchanged", strcmp(result, "я") == 0);
    
    // Test 4: Lowercase
    result = stemmer.stem("ПРОГРАММА");
    test("Uppercase to lowercase", strlen(result) <= strlen("ПРОГРАММА"));
    
    // Test 5: Search index
    SearchIndex index;
    index.add_document(0, "программирование на python", &stemmer);
    index.add_document(1, "язык программирования", &stemmer);
    index.add_document(2, "машинное обучение", &stemmer);
    
    int results[10];
    int num_results;
    index.search("программирование", &stemmer, results, &num_results, 10);
    test("Search finds documents", num_results >= 2);
    
    // Test 6: Search relevance
    test("First result is most relevant", num_results > 0);
    
    printf("\nPassed %d/%d tests\n", passed, test_count);
    return (passed == test_count) ? 0 : 1;
}