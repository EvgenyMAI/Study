#ifndef STEMMER_H
#define STEMMER_H

#include <cstring>
#include <cctype>

class RussianStemmer {
private:
    char* word;
    int len;
    int rv_pos;
    int r1_pos;
    int r2_pos;
    
    // UTF-8 helpers
    int utf8_strlen(const char* str);
    void utf8_to_lower(char* str);
    bool is_vowel(const char* pos);
    
    // Region calculation
    void calculate_regions();
    
    // Suffix removal
    bool ends_with(const char* suffix, int from_pos = 0);
    void remove_ending(int chars_to_remove);
    bool try_remove_perfective_gerund();
    bool try_remove_adjective();
    bool try_remove_reflexive();
    bool try_remove_verb();
    bool try_remove_noun();
    bool try_remove_i();
    bool try_remove_derivational();
    void try_remove_superlative_and_nn();
    void try_remove_soft_sign();
    
public:
    RussianStemmer();
    ~RussianStemmer();
    char* stem(const char* input);
};

#endif