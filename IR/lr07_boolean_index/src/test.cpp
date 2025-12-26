#include <cstdio>
#include <cstring>
#include "posting_list.h"
#include "boolean_index.h"
#include "boolean_query.h"

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
    printf("Testing Boolean Index (Optimized Version)...\n\n");

    // Test 1: PostingList - add and contains
    PostingList list1;
    list1.add_document(1);
    list1.add_document(3);
    list1.add_document(5);
    list1.finalize();
    test("PostingList add and contains", 
         list1.contains(1) && list1.contains(3) && list1.contains(5) && !list1.contains(2));

    // Test 2: PostingList - no duplicates after finalize
    PostingList list_dup;
    list_dup.add_document(3);
    list_dup.add_document(3);
    list_dup.add_document(5);
    list_dup.finalize();
    test("PostingList no duplicates", list_dup.get_size() == 2);

    // Test 3: PostingList - fast copy
    PostingList* list1_copy = list1.copy();
    test("PostingList fast copy",
         list1_copy->get_size() == list1.get_size() &&
         list1_copy->contains(1) && list1_copy->contains(3) && list1_copy->contains(5));
    delete list1_copy;

    // Test 4: PostingList - AND operation
    PostingList list2;
    list2.add_document(2);
    list2.add_document(3);
    list2.add_document(4);
    list2.finalize();

    PostingList* result_and = PostingList::intersect(&list1, &list2);
    test("PostingList AND operation", 
         result_and->get_size() == 1 && result_and->contains(3));
    delete result_and;

    // Test 5: PostingList - OR operation
    PostingList* result_or = PostingList::union_lists(&list1, &list2);
    test("PostingList OR operation", 
         result_or->get_size() == 5 && result_or->contains(1) && 
         result_or->contains(2) && result_or->contains(3) && 
         result_or->contains(4) && result_or->contains(5));
    delete result_or;

    // Test 6: PostingList - NOT operation
    PostingList* result_not = PostingList::difference(&list1, &list2);
    test("PostingList NOT operation", 
         result_not->get_size() == 2 && result_not->contains(1) && result_not->contains(5));
    delete result_not;

    // Test 7: BooleanIndex - add and retrieve
    BooleanIndex index;
    index.add_term("test", 0);
    index.add_term("test", 1);
    index.add_term("hello", 0);
    index.finalize();

    PostingList* postings = index.get_postings("test");
    test("BooleanIndex add and retrieve", 
         postings && postings->get_size() == 2 && 
         postings->contains(0) && postings->contains(1));

    // Test 8: BooleanIndex - multiple terms
    test("BooleanIndex multiple terms", index.get_num_terms() == 2);

    // Test 9: BooleanQuery - simple query
    index.add_term("world", 1);
    index.add_term("world", 2);
    index.set_num_docs(3);
    index.finalize();

    BooleanQuery bq(&index);
    PostingList* query_result = bq.execute("test AND world");
    test("BooleanQuery simple AND", 
         query_result->get_size() == 1 && query_result->contains(1));
    delete query_result;

    // Test 10: BooleanQuery - OR query
    query_result = bq.execute("hello OR world");
    test("BooleanQuery simple OR", 
         query_result->get_size() == 3 && query_result->contains(0) && 
         query_result->contains(1) && query_result->contains(2));
    delete query_result;

    // Test 11: Performance test - массовая вставка
    printf("\nPerformance test: adding 10000 documents...\n");
    BooleanIndex perf_index;
    for (int i = 0; i < 10000; i++) {
        perf_index.add_term("word", i);
        if (i % 1000 == 0 && i > 0) {
            printf("Added %d documents...\r", i);
            fflush(stdout);
        }
    }
    perf_index.finalize();
    printf("\nAdded 10000 documents successfully!\n");
    PostingList* perf_postings = perf_index.get_postings("word");
    test("Performance test", perf_postings && perf_postings->get_size() == 10000);

    // Test 12: Copy performance test
    printf("\nPerformance test: copying large posting list...\n");
    PostingList* perf_copy = perf_postings->copy();
    test("Fast copy performance test", 
         perf_copy && perf_copy->get_size() == 10000 &&
         perf_copy->contains(0) && perf_copy->contains(9999));
    delete perf_copy;
    printf("Copy completed successfully!\n");

    printf("\n=========================\n");
    printf("Passed %d/%d tests\n", passed, test_count);
    printf("=========================\n");
    return (passed == test_count) ? 0 : 1;
}