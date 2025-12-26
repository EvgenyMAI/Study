#include "posting_list.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

PostingList::PostingList() : documents(nullptr), capacity(0), size(0), is_sorted(false) {}

PostingList::~PostingList() {
    if (documents) {
        free(documents);
    }
}

void PostingList::resize() {
    int new_capacity = (capacity == 0) ? 10 : capacity * 2;
    documents = (int*)realloc(documents, new_capacity * sizeof(int));
    capacity = new_capacity;
}

void PostingList::add_document(int doc_id) {
    // Быстрая вставка в конец без проверок - O(1)
    if (size >= capacity) {
        resize();
    }
    documents[size++] = doc_id;
    is_sorted = false;  // Пометить как несортированный
}

void PostingList::sort_and_deduplicate() {
    if (size == 0) {
        is_sorted = true;
        return;
    }

    // Сортировка массива - O(n log n)
    std::sort(documents, documents + size);

    // Удаление дубликатов - O(n)
    int write_pos = 0;
    for (int read_pos = 0; read_pos < size; read_pos++) {
        if (read_pos == 0 || documents[read_pos] != documents[read_pos - 1]) {
            documents[write_pos++] = documents[read_pos];
        }
    }
    size = write_pos;

    is_sorted = true;
}

void PostingList::finalize() {
    if (!is_sorted) {
        sort_and_deduplicate();
    }
}

// Быстрое копирование через memcpy - O(n)
PostingList* PostingList::copy() const {
    PostingList* new_list = new PostingList();

    if (size > 0) {
        // Выделяем память сразу под все элементы
        new_list->capacity = size;
        new_list->size = size;
        new_list->documents = (int*)malloc(size * sizeof(int));

        // Быстрое копирование памяти
        memcpy(new_list->documents, documents, size * sizeof(int));
        new_list->is_sorted = is_sorted;
    }

    return new_list;
}

bool PostingList::contains(int doc_id) const {
    if (!is_sorted) {
        // Линейный поиск если не отсортирован
        for (int i = 0; i < size; i++) {
            if (documents[i] == doc_id) return true;
        }
        return false;
    }

    // Бинарный поиск если отсортирован - O(log n)
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (documents[mid] == doc_id) return true;
        if (documents[mid] < doc_id) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return false;
}

int PostingList::get_size() const {
    return size;
}

int PostingList::get_document(int index) const {
    if (index >= 0 && index < size) {
        return documents[index];
    }
    return -1;
}

void PostingList::save(FILE* f) const {
    fprintf(f, "%d\n", size);
    for (int i = 0; i < size; i++) {
        fprintf(f, "%d ", documents[i]);
    }
    fprintf(f, "\n");
}

void PostingList::load(FILE* f) {
    int n;
    fscanf(f, "%d\n", &n);

    // Выделяем память сразу под все элементы
    if (n > capacity) {
        documents = (int*)realloc(documents, n * sizeof(int));
        capacity = n;
    }

    for (int i = 0; i < n; i++) {
        fscanf(f, "%d ", &documents[i]);
    }
    size = n;
    is_sorted = true;  // Предполагаем что загруженные данные отсортированы
}

// AND: пересечение отсортированных списков - O(n + m)
PostingList* PostingList::intersect(const PostingList* list1, const PostingList* list2) {
    PostingList* result = new PostingList();

    if (!list1 || !list2 || list1->size == 0 || list2->size == 0) {
        return result;
    }

    int i = 0, j = 0;
    while (i < list1->size && j < list2->size) {
        if (list1->documents[i] == list2->documents[j]) {
            result->add_document(list1->documents[i]);
            i++;
            j++;
        } else if (list1->documents[i] < list2->documents[j]) {
            i++;
        } else {
            j++;
        }
    }

    result->is_sorted = true;  // Результат уже отсортирован
    return result;
}

// OR: объединение отсортированных списков - O(n + m)
PostingList* PostingList::union_lists(const PostingList* list1, const PostingList* list2) {
    PostingList* result = new PostingList();

    if (!list1 && !list2) return result;
    if (!list1 || list1->size == 0) {
        return list2->copy();
    }
    if (!list2 || list2->size == 0) {
        return list1->copy();
    }

    int i = 0, j = 0;
    while (i < list1->size && j < list2->size) {
        if (list1->documents[i] == list2->documents[j]) {
            result->add_document(list1->documents[i]);
            i++;
            j++;
        } else if (list1->documents[i] < list2->documents[j]) {
            result->add_document(list1->documents[i]);
            i++;
        } else {
            result->add_document(list2->documents[j]);
            j++;
        }
    }

    while (i < list1->size) {
        result->add_document(list1->documents[i++]);
    }

    while (j < list2->size) {
        result->add_document(list2->documents[j++]);
    }

    result->is_sorted = true;  // Результат уже отсортирован
    return result;
}

// NOT: разность списков (list1 - list2) - O(n + m)
PostingList* PostingList::difference(const PostingList* list1, const PostingList* list2) {
    PostingList* result = new PostingList();

    if (!list1 || list1->size == 0) return result;
    if (!list2 || list2->size == 0) {
        return list1->copy();
    }

    int i = 0, j = 0;
    while (i < list1->size && j < list2->size) {
        if (list1->documents[i] < list2->documents[j]) {
            result->add_document(list1->documents[i]);
            i++;
        } else if (list1->documents[i] == list2->documents[j]) {
            i++;
            j++;
        } else {
            j++;
        }
    }

    while (i < list1->size) {
        result->add_document(list1->documents[i++]);
    }

    result->is_sorted = true;  // Результат уже отсортирован
    return result;
}

void PostingList::print() const {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", documents[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]");
}