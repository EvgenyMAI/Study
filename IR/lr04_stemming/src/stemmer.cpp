#include "stemmer.h"
#include <cstdlib>

RussianStemmer::RussianStemmer() : word(nullptr), len(0), rv_pos(0), r1_pos(0), r2_pos(0) {}

RussianStemmer::~RussianStemmer() {
    if (word) {
        free(word);
    }
}

int RussianStemmer::utf8_strlen(const char* str) {
    int count = 0;
    while (*str) {
        if ((*str & 0xC0) != 0x80) count++;
        str++;
    }
    return count;
}

void RussianStemmer::utf8_to_lower(char* str) {
    unsigned char* ustr = (unsigned char*)str;
    int i = 0;
    while (ustr[i]) {
        // Russian А-Я (0xD090-0xD0AF) -> а-я (0xD0B0-0xD0CF)
        if (ustr[i] == 0xD0 && ustr[i+1] >= 0x90 && ustr[i+1] <= 0x9F) {
            ustr[i+1] += 0x20;
            i += 2;
        }
        // Russian А-П (0xD090-0xD09F) already handled above
        // Russian Р-Я (0xD0A0-0xD0AF) -> р-я (0xD180-0xD18F)
        else if (ustr[i] == 0xD0 && ustr[i+1] >= 0xA0 && ustr[i+1] <= 0xAF) {
            ustr[i] = 0xD1;
            ustr[i+1] -= 0x20;
            i += 2;
        }
        // Latin A-Z -> a-z
        else if (ustr[i] >= 'A' && ustr[i] <= 'Z') {
            ustr[i] += 32;
            i++;
        }
        // Ё (0xD001) -> е (0xD0B5)
        else if (ustr[i] == 0xD0 && ustr[i+1] == 0x81) {
            ustr[i+1] = 0xB5;
            i += 2;
        }
        else {
            i++;
        }
    }
}

bool RussianStemmer::is_vowel(const char* pos) {
    if (!pos || *pos == '\0') return false;
    unsigned char c1 = (unsigned char)pos[0];
    unsigned char c2 = (unsigned char)pos[1];
    
    // Russian vowels: а, е, и, о, у, ы, э, ю, я
    if (c1 == 0xD0) {
        return (c2 == 0xB0 || c2 == 0xB5 || c2 == 0xB8 || c2 == 0xBE || 
                c2 == 0xBF || c2 == 0xBC || c2 == 0xBD);
    }
    if (c1 == 0xD1) {
        return (c2 == 0x8B || c2 == 0x8D || c2 == 0x8E || c2 == 0x8F);
    }
    // English vowels: a, e, i, o, u, y
    if (c1 < 0x80) {
        return (c1 == 'a' || c1 == 'e' || c1 == 'i' || 
                c1 == 'o' || c1 == 'u' || c1 == 'y');
    }
    return false;
}

void RussianStemmer::calculate_regions() {
    rv_pos = 0;
    r1_pos = len;
    r2_pos = len;
    
    // Find RV (after first vowel)
    const char* p = word;
    bool found_vowel = false;
    int pos = 0;
    while (*p) {
        if (is_vowel(p)) {
            found_vowel = true;
            // Move past the vowel
            if ((*p & 0x80) != 0) {
                p += 2;
                pos += 2;
            } else {
                p++;
                pos++;
            }
            rv_pos = pos;
            break;
        }
        if ((*p & 0x80) != 0) {
            p += 2;
            pos += 2;
        } else {
            p++;
            pos++;
        }
    }
    
    // Find R1 (after first non-vowel following a vowel)
    if (found_vowel) {
        bool in_vowel_seq = true;
        while (*p) {
            bool curr_is_vowel = is_vowel(p);
            if (in_vowel_seq && !curr_is_vowel) {
                in_vowel_seq = false;
            } else if (!in_vowel_seq && !curr_is_vowel) {
                if ((*p & 0x80) != 0) {
                    pos += 2;
                    p += 2;
                } else {
                    pos++;
                    p++;
                }
                r1_pos = pos;
                break;
            } else if (!in_vowel_seq && curr_is_vowel) {
                in_vowel_seq = true;
            }
            if ((*p & 0x80) != 0) {
                pos += 2;
                p += 2;
            } else {
                pos++;
                p++;
            }
        }
    }
    
    // Find R2 (same logic from R1)
    if (r1_pos < len) {
        p = word + r1_pos;
        pos = r1_pos;
        bool in_vowel_seq = is_vowel(p);
        while (*p) {
            bool curr_is_vowel = is_vowel(p);
            if (in_vowel_seq && !curr_is_vowel) {
                in_vowel_seq = false;
            } else if (!in_vowel_seq && !curr_is_vowel) {
                if ((*p & 0x80) != 0) {
                    pos += 2;
                    p += 2;
                } else {
                    pos++;
                    p++;
                }
                r2_pos = pos;
                break;
            } else if (!in_vowel_seq && curr_is_vowel) {
                in_vowel_seq = true;
            }
            if ((*p & 0x80) != 0) {
                pos += 2;
                p += 2;
            } else {
                pos++;
                p++;
            }
        }
    }
}

bool RussianStemmer::ends_with(const char* suffix, int from_pos) {
    int suffix_len = strlen(suffix);
    int start = (from_pos == 0) ? rv_pos : from_pos;
    
    if (len < start + suffix_len) return false;
    
    return strcmp(word + len - suffix_len, suffix) == 0;
}

void RussianStemmer::remove_ending(int bytes) {
    if (len >= bytes) {
        len -= bytes;
        word[len] = '\0';
    }
}

bool RussianStemmer::try_remove_perfective_gerund() {
    // в, вши, вшись
    if (ends_with("\xD0\xB2")) { // в
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xB2\xD1\x88\xD0\xB8")) { // вши
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB2\xD1\x88\xD0\xB8\xD1\x81\xD1\x8C")) { // вшись
        remove_ending(10);
        return true;
    }
    
    // ив, ивши, ившись, ыв, ывши, ывшись
    if (ends_with("\xD0\xB8\xD0\xB2")) { // ив
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB2\xD1\x88\xD0\xB8")) { // ивши
        remove_ending(8);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB2\xD1\x88\xD0\xB8\xD1\x81\xD1\x8C")) { // ившись
        remove_ending(12);
        return true;
    }
    if (ends_with("\xD1\x8B\xD0\xB2")) { // ыв
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8B\xD0\xB2\xD1\x88\xD0\xB8")) { // ывши
        remove_ending(8);
        return true;
    }
    if (ends_with("\xD1\x8B\xD0\xB2\xD1\x88\xD0\xB8\xD1\x81\xD1\x8C")) { // ывшись
        remove_ending(12);
        return true;
    }
    
    return false;
}

bool RussianStemmer::try_remove_reflexive() {
    // ся, сь
    if (ends_with("\xD1\x81\xD1\x8F")) { // ся
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x81\xD1\x8C")) { // сь
        remove_ending(4);
        return true;
    }
    return false;
}

bool RussianStemmer::try_remove_adjective() {
    // ее, ие, ые, ое, ими, ыми, ей, ий, ый, ой, ем, им, ым, ом, его, ого, ему, ому, их, ых, ую, юю, ая, яя, ою, ею
    const char* adj_endings[] = {
        "\xD0\xB5\xD0\xB5", "\xD0\xB8\xD0\xB5", "\xD1\x8B\xD0\xB5", "\xD0\xBE\xD0\xB5",
        "\xD0\xB8\xD0\xBC\xD0\xB8", "\xD1\x8B\xD0\xBC\xD0\xB8",
        "\xD0\xB5\xD0\xB9", "\xD0\xB8\xD0\xB9", "\xD1\x8B\xD0\xB9", "\xD0\xBE\xD0\xB9",
        "\xD0\xB5\xD0\xBC", "\xD0\xB8\xD0\xBC", "\xD1\x8B\xD0\xBC", "\xD0\xBE\xD0\xBC",
        "\xD0\xB5\xD0\xB3\xD0\xBE", "\xD0\xBE\xD0\xB3\xD0\xBE",
        "\xD0\xB5\xD0\xBC\xD1\x83", "\xD0\xBE\xD0\xBC\xD1\x83",
        "\xD0\xB8\xD1\x85", "\xD1\x8B\xD1\x85",
        "\xD1\x83\xD1\x8E", "\xD1\x8E\xD1\x8E",
        "\xD0\xB0\xD1\x8F", "\xD1\x8F\xD1\x8F",
        "\xD0\xBE\xD1\x8E", "\xD0\xB5\xD1\x8E",
        nullptr
    };
    
    for (int i = 0; adj_endings[i] != nullptr; i++) {
        if (ends_with(adj_endings[i])) {
            int suffix_len = strlen(adj_endings[i]);
            remove_ending(suffix_len);
            return true;
        }
    }
    return false;
}

bool RussianStemmer::try_remove_verb() {
    // ла, на, ете, йте, ли, й, л, ем, н, ло, но, ет, ют, ны, ть, ешь, нно
    const char* verb_endings[] = {
        "\xD0\xBB\xD0\xB0", "\xD0\xBD\xD0\xB0",
        "\xD0\xB5\xD1\x82\xD0\xB5",
        "\xD0\xB9\xD1\x82\xD0\xB5",
        "\xD0\xBB\xD0\xB8", "\xD0\xB9", "\xD0\xBB",
        "\xD0\xB5\xD0\xBC", "\xD0\xBD",
        "\xD0\xBB\xD0\xBE", "\xD0\xBD\xD0\xBE",
        "\xD0\xB5\xD1\x82",
        "\xD1\x8E\xD1\x82",
        "\xD0\xBD\xD1\x8B",
        "\xD1\x82\xD1\x8C",
        "\xD0\xB5\xD1\x88\xD1\x8C",
        "\xD0\xBD\xD0\xBD\xD0\xBE",
        nullptr
    };
    
    for (int i = 0; verb_endings[i] != nullptr; i++) {
        if (ends_with(verb_endings[i])) {
            int suffix_len = strlen(verb_endings[i]);
            remove_ending(suffix_len);
            return true;
        }
    }
    
    // ила, ыла, ена, ите, или, ыли, ей, уй, ил, ыл, им, ым, ен, ило, ыло, ено, ят, ует, уют, ит, ыт, ены, ить, ыть, ишь, ую, ю
    const char* verb_endings2[] = {
        "\xD0\xB8\xD0\xBB\xD0\xB0", "\xD1\x8B\xD0\xBB\xD0\xB0",
        "\xD0\xB5\xD0\xBD\xD0\xB0",
        "\xD0\xB8\xD1\x82\xD0\xB5",
        "\xD0\xB8\xD0\xBB\xD0\xB8", "\xD1\x8B\xD0\xBB\xD0\xB8",
        "\xD0\xB5\xD0\xB9", "\xD1\x83\xD0\xB9",
        "\xD0\xB8\xD0\xBB", "\xD1\x8B\xD0\xBB",
        "\xD0\xB8\xD0\xBC", "\xD1\x8B\xD0\xBC",
        "\xD0\xB5\xD0\xBD",
        "\xD0\xB8\xD0\xBB\xD0\xBE", "\xD1\x8B\xD0\xBB\xD0\xBE",
        "\xD0\xB5\xD0\xBD\xD0\xBE",
        "\xD1\x8F\xD1\x82",
        "\xD1\x83\xD0\xB5\xD1\x82", "\xD1\x83\xD1\x8E\xD1\x82",
        "\xD0\xB8\xD1\x82", "\xD1\x8B\xD1\x82",
        "\xD0\xB5\xD0\xBD\xD1\x8B",
        "\xD0\xB8\xD1\x82\xD1\x8C", "\xD1\x8B\xD1\x82\xD1\x8C",
        "\xD0\xB8\xD1\x88\xD1\x8C",
        "\xD1\x83\xD1\x8E", "\xD1\x8E",
        nullptr
    };
    
    for (int i = 0; verb_endings2[i] != nullptr; i++) {
        if (ends_with(verb_endings2[i])) {
            int suffix_len = strlen(verb_endings2[i]);
            remove_ending(suffix_len);
            return true;
        }
    }
    
    return false;
}

bool RussianStemmer::try_remove_noun() {
    // Важно: проверяем более длинные окончания первыми!
    
    // 8 байт
    if (ends_with("\xD0\xB8\xD1\x8F\xD0\xBC\xD0\xB8")) { // иями
        remove_ending(8);
        return true;
    }
    
    // 6 байт
    if (ends_with("\xD1\x8F\xD0\xBC\xD0\xB8")) { // ями
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB0\xD0\xBC\xD0\xB8")) { // ами
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB8\xD1\x8F\xD1\x85")) { // иях
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB5\xD0\xB9")) { // ией
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB8\xD1\x8F\xD0\xBC")) { // иям
        remove_ending(6);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB5\xD0\xBC")) { // ием
        remove_ending(6);
        return true;
    }
    
    // 4 байта - ВАЖНО: "ов" и "ев" проверяем ДО "ов"
    if (ends_with("\xD0\xBE\xD0\xB2")) { // ов
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB5\xD0\xB2")) { // ев
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8F\xD1\x85")) { // ях
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB0\xD1\x85")) { // ах
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB5\xD0\xB9")) { // ей
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xBE\xD0\xB9")) { // ой
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB9")) { // ий
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8F\xD0\xBC")) { // ям
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB5\xD0\xBC")) { // ем
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB0\xD0\xBC")) { // ам
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xBE\xD0\xBC")) { // ом
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD1\x8E")) { // ию
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8C\xD1\x8E")) { // ью
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD1\x8F")) { // ия
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8C\xD1\x8F")) { // ья
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB5")) { // ие
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD1\x8C\xD0\xB5")) { // ье
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB5\xD0\xB8")) { // еи
        remove_ending(4);
        return true;
    }
    if (ends_with("\xD0\xB8\xD0\xB8")) { // ии
        remove_ending(4);
        return true;
    }
    
    // 2 байта
    if (ends_with("\xD1\x8E")) { // ю
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD1\x8F")) { // я
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xB5")) { // е
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xB8")) { // и
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xB9")) { // й
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xBE")) { // о
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD1\x83")) { // у
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD1\x8B")) { // ы
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD1\x8C")) { // ь
        remove_ending(2);
        return true;
    }
    if (ends_with("\xD0\xB0")) { // а
        remove_ending(2);
        return true;
    }
    
    return false;
}

bool RussianStemmer::try_remove_i() {
    // и
    if (ends_with("\xD0\xB8")) {
        remove_ending(2);
        return true;
    }
    return false;
}

bool RussianStemmer::try_remove_derivational() {
    // ость, ост in R2
    if (len >= r2_pos + 8 && ends_with("\xD0\xBE\xD1\x81\xD1\x82\xD1\x8C")) { // ость
        remove_ending(8);
        return true;
    }
    if (len >= r2_pos + 6 && ends_with("\xD0\xBE\xD1\x81\xD1\x82")) { // ост
        remove_ending(6);
        return true;
    }
    return false;
}

void RussianStemmer::try_remove_superlative_and_nn() {
    // ейш, ейше
    if (ends_with("\xD0\xB5\xD0\xB9\xD1\x88\xD0\xB5")) { // ейше
        remove_ending(8);
    } else if (ends_with("\xD0\xB5\xD0\xB9\xD1\x88")) { // ейш
        remove_ending(6);
    }
    
    // Remove double н
    if (len >= 4 && word[len-2] == '\xD0' && word[len-1] == '\xBD' &&
        word[len-4] == '\xD0' && word[len-3] == '\xBD') { // нн
        remove_ending(2);
    }
}

void RussianStemmer::try_remove_soft_sign() {
    // ь
    if (ends_with("\xD1\x8C")) {
        remove_ending(2);
    }
}

char* RussianStemmer::stem(const char* input) {
    if (word) {
        free(word);
    }
    
    word = (char*)malloc(strlen(input) + 1);
    strcpy(word, input);
    utf8_to_lower(word);
    len = strlen(word);
    
    if (len < 4) { // Minimum word length
        return word;
    }
    
    calculate_regions();
    
    // Step 1
    if (!try_remove_perfective_gerund()) {
        try_remove_reflexive();
        if (!try_remove_adjective()) {
            if (!try_remove_verb()) {
                try_remove_noun();
            }
        }
    }
    
    // Step 2
    try_remove_i();
    
    // Step 3
    try_remove_derivational();
    
    // Step 4
    try_remove_superlative_and_nn();
    try_remove_soft_sign();
    
    return word;
}