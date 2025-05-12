import re
from gmpy2 import mpz, gcd
from pathlib import Path

# Чтение содержимого файла
def read_file(filename):
    try:
        return Path(filename).read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"Ошибка: не удалось открыть файл {filename}")
        exit(1)

# Извлечение чисел B с помощью регулярного выражения
def extract_numbers(text):
    pattern = re.compile(r"b\[\d+\]=([\d\s]+)")
    numbers = []
    for match in pattern.finditer(text):
        num_str = re.sub(r"\s+", "", match.group(1))
        numbers.append(mpz(num_str))
    return numbers

# Поиск общих делителей
def find_common_divisors(target, candidates, max_count=2):
    result = []
    for num in candidates:
        common = gcd(target, num)
        if common > 1 and common != target:
            result.append(common)
            result.append(target // common)
            if len(result) >= max_count:
                break
    return result

def main():
    filename = "lab2.txt"
    file_content = read_file(filename)
    numbers = extract_numbers(file_content)

    my_b = mpz("32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463688717960921898019494119559150490921095088154634506699037027779726363798979981367160679925750849971537538870855219307801448348596921540627795219927951379004452444719570738042378337465944112665294993327737106405440544513705275406544717522899690555069561839963232186804956513836192717374145392828808695477033015558202787064995031407079312795125272223392253120970835275756415864155319445535167341795525293598484555252724818812724051787201347021168925285903218700413299571102718214809898535950307826378982148851017973389")

    divisors = find_common_divisors(my_b, numbers)

    for d in divisors:
        print("Общий делитель найден:", d)

if __name__ == "__main__":
    main()