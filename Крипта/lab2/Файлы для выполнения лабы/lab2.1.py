import sys
from gostcrypto import gosthash

def get_streebog_hash(data: bytes) -> bytes:
    """Вычисляет хеш Streebog256 по переданным байтам."""
    hasher = gosthash.new('streebog256')
    hasher.update(data)
    return hasher.digest()

def calculate_variant_from_fio(fio: str) -> tuple[str, int]:
    """Вычисляет номер варианта на основе хеша ФИО."""
    byte_data = fio.encode('utf-8')
    hash_result = get_streebog_hash(byte_data)
    variant_number = hash_result[-1]
    return hash_result.hex(), variant_number

def main():
    fio_input = "Кострюков Евгений Сергеевич"
    hash_hex, variant = calculate_variant_from_fio(fio_input)
    print("ФИО:", fio_input)
    print("Streebog256 хеш:", hash_hex)
    print(f"Номер варианта: {variant} (HEX: 0x{variant:02X}, DEC: {variant})")

if __name__ == "__main__":
    main()