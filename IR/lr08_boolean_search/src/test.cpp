#include <cstdio>
#include "../../lr07_boolean_index/src/boolean_index.h"
#include "search_engine.h"

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
    printf("Testing Boolean Search Engine...\n\n");
    
    // Создаем тестовый индекс
    BooleanIndex index;
    
    // Документ 0: "python программирование"
    index.add_term("python", 0);
    index.add_term("программирование", 0);
    
    // Документ 1: "python java"
    index.add_term("python", 1);
    index.add_term("java", 1);
    
    // Документ 2: "java программирование"
    index.add_term("java", 2);
    index.add_term("программирование", 2);
    
    // Документ 3: "машинное обучение"
    index.add_term("машинное", 3);
    index.add_term("обучение", 3);
    
    index.set_num_docs(4);
    
    SearchEngine engine(&index);
    
    // Test 1: Простой запрос
    PostingList* result = engine.search("python");
    test("Simple query 'python'", result->get_size() == 2);
    delete result;
    
    // Test 2: AND запрос
    result = engine.search("python AND программирование");
    test("AND query", result->get_size() == 1 && result->contains(0));
    delete result;
    
    // Test 3: OR запрос
    result = engine.search("python OR машинное");
    test("OR query", result->get_size() == 3);
    delete result;
    
    // Test 4: NOT запрос
    result = engine.search("python AND NOT java");
    test("NOT query", result->get_size() == 1 && result->contains(0));
    delete result;
    
    // Test 5: Комбинированный запрос
    result = engine.search("python OR java");
    test("Combined query", result->get_size() == 3);
    delete result;
    
    // Test 6: Пустой результат
    result = engine.search("несуществующее");
    test("Empty result", result->get_size() == 0);
    delete result;
    
    // Test 7: Неявный AND (два термина подряд)
    result = engine.search("python программирование");
    test("Implicit AND", result->get_size() == 1);
    delete result;
    
    printf("\nPassed %d/%d tests\n", passed, test_count);
    return (passed == test_count) ? 0 : 1;
}