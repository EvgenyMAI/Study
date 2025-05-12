import os

FOLDER = os.path.dirname(os.path.abspath(__file__))
TEXT_LENGTH = 5000

def load_text(filename):
    with open(os.path.join(FOLDER, filename), "r", encoding="utf-8") as f:
        return f.read()[:TEXT_LENGTH]

def compare_texts(text1, text2):
    return sum(c1 == c2 for c1, c2 in zip(text1, text2)) / len(text1)

if __name__ == "__main__":
    # Загрузка всех 6 текстов
    t1 = load_text("meaningful_1.txt")
    t2 = load_text("meaningful_2.txt")
    rl1 = load_text("random_letters_1.txt")
    rl2 = load_text("random_letters_2.txt")
    rw1 = load_text("random_words_1.txt")
    rw2 = load_text("random_words_2.txt")

    comparisons = [
        ("Осмысленный 1 vs Осмысленный 2", t1, t2),
        ("Осмысленный 1 vs Случайные буквы 1", t1, rl1),
        ("Осмысленный 2 vs Случайные слова 2", t2, rw2),
        ("Случайные буквы 1 vs буквы 2", rl1, rl2),
        ("Случайные слова 1 vs слова 2", rw1, rw2),
    ]

    print("Результаты сравнения (доля совпадающих символов):")
    for name, a, b in comparisons:
        score = compare_texts(a, b)
        print(f"{name:<40} --> {score:.4f}")