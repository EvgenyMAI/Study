#ifndef POSTING_LIST_H
#define POSTING_LIST_H

#include <cstdio>

// Список документов для одного термина (posting list)
class PostingList {
private:
    int* documents;      // Динамический массив ID документов
    int capacity;        // Вместимость массива
    int size;            // Текущий размер
    bool is_sorted;      // Флаг сортировки

    void resize();       // Увеличение capacity
    void sort_and_deduplicate();  // Сортировка + удаление дубликатов

public:
    PostingList();
    ~PostingList();

    void add_document(int doc_id);
    bool contains(int doc_id) const;
    int get_size() const;
    int get_document(int index) const;  // Получить документ по индексу

    PostingList* copy() const;

    // Подготовка к использованию (сортировка)
    void finalize();

    // Сохранение/загрузка
    void save(FILE* f) const;
    void load(FILE* f);

    // Операции над списками
    static PostingList* intersect(const PostingList* list1, const PostingList* list2); // AND
    static PostingList* union_lists(const PostingList* list1, const PostingList* list2); // OR
    static PostingList* difference(const PostingList* list1, const PostingList* list2); // NOT

    // Вывод для отладки
    void print() const;
};

#endif