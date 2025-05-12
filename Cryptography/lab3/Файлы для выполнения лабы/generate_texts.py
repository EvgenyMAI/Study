import os
import random
import string

FOLDER = os.path.dirname(os.path.abspath(__file__))
WORDS_FILE = os.path.join(FOLDER, "1000-most-common-words.txt")
TEXT_LENGTH = 5000

def load_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def generate_random_words_text(words_list, length):
    text = ""
    while len(text) < length:
        text += random.choice(words_list) + " "
    return text[:length]

def generate_random_letters_text(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def save_text(filename, text):
    with open(os.path.join(FOLDER, filename), "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    words = load_words(WORDS_FILE)

    random_words_text_1 = generate_random_words_text(words, TEXT_LENGTH)
    random_words_text_2 = generate_random_words_text(words, TEXT_LENGTH)
    random_letters_text_1 = generate_random_letters_text(TEXT_LENGTH)
    random_letters_text_2 = generate_random_letters_text(TEXT_LENGTH)

    save_text("random_words_1.txt", random_words_text_1)
    save_text("random_words_2.txt", random_words_text_2)
    save_text("random_letters_1.txt", random_letters_text_1)
    save_text("random_letters_2.txt", random_letters_text_2)

    print("4 файла успешно сгенерированы и сохранены.")