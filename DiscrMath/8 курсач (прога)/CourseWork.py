from tkinter import messagebox, scrolledtext
from PIL import ImageTk, Image
import tkinter as tk
import math
import os

def window_of_decoding():
    words = input_entry.get().split()
    encoded_words = []
    for word in words:
        msg_bit = [int(bit) for bit in word]
        try:
            hamming_code = code_of_hamming(msg_bit)
            encoded_words.append(''.join(map(str, hamming_code)))
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
    
    result_window = tk.Toplevel(window)
    result_window.title("Результат кодирования")
    result_window.geometry("300x200")
    result_text = scrolledtext.ScrolledText(result_window, width=30, height=10)

def code_of_hamming(chet_bit):
    m = len(chet_bit)
    r = 1
    while 2 ** r < (m + r + 1):
        r += 1

    hamming_code = [-1] * (r + m)
    j = 0
    for i in range(r + m):
        if math.log2(i + 1).is_integer():
            hamming_code[i] = 0
        else:
            hamming_code[i] = chet_bit[j]
            j += 1
    for i in range(r):
        parity_index = 2 ** i - 1
        count = 0
        for j in range(parity_index, len(hamming_code), 2 * (parity_index + 1)):
            for k in range(parity_index + 1):
                if j + k >= len(hamming_code):
                    break
                count += hamming_code[j + k]

        hamming_code[parity_index] = 1 if count % 2 == 1 else 0
    return hamming_code

def print_sg(data):
    m = len(data)
    r = 1
    while 2 ** r < (m + r + 1):
        r += 1
    hamming_code = code_of_hamming(data)
    print("Message bits are:", data)
    print("Hamming code is:", hamming_code)
    
def code_decoding(hamming_code):
    r = 1
    while 2 ** r < len(hamming_code):
        r += 1
    error_position = 0
    for i in range(r):
        parity_index = 2 ** i - 1
        count = 0
        for j in range(parity_index, len(hamming_code), 2 * (parity_index + 1)):
            for k in range(parity_index + 1):
                if j + k >= len(hamming_code):
                    break
                count += hamming_code[j + k]
        if count % 2 != 0:
            error_position += parity_index + 1
    if error_position > 0:
        hamming_code[error_position - 1] = (hamming_code[error_position - 1] + 1) % 2
    decoded_msg_bit = []
    j = 0
    for i in range(len(hamming_code)):
        if not math.log2(i + 1).is_integer():
            decoded_msg_bit.append(hamming_code[i])
            j += 1
    return decoded_msg_bit

def window_of_decoding():
    words = input_entry.get().split()
    encoded_words = []
    for word in words:
        msg_bit = [int(bit) for bit in word]
        try:
            hamming_code = code_of_hamming(msg_bit)
            encoded_words.append(''.join(map(str, hamming_code)))
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
    result_window = tk.Toplevel(window)
    result_window.title("Результат кодирования")
    result_window.geometry("300x200")
    result_text = scrolledtext.ScrolledText(result_window, width=30, height=10)
    result_text.insert(tk.INSERT, '\n'.join(encoded_words))
    result_text.pack()

def decode_button_click():
    hamming_codes = input_entry.get().split()
    decoded_words = []
    for code in hamming_codes:
        hamming_code = [int(bit) for bit in code]
        try:
            decoded_msg_bit = code_decoding(hamming_code)
            decoded_words.append(''.join(map(str, decoded_msg_bit)))
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
    result_window = tk.Toplevel(window)
    result_window.title("Результат декодирования")
    result_window.geometry("300x200")
    result_text = scrolledtext.ScrolledText(result_window, width=30, height=10)
    result_text.insert(tk.INSERT, ' '.join(decoded_words))
    result_text.pack()

# Создаём окно
window = tk.Tk()
window.title("Кодирование и декодирование кода Хемминга")
window.geometry("640x480")

# Получаем путь к файлу программы
current_dir = os.path.dirname(os.path.abspath(__file__))

# Определяем путь к изображению на фоне
image_path = os.path.join(current_dir, "back2.jpg")

# Загружаем изображения
image = Image.open(image_path)
background_image = ImageTk.PhotoImage(image)

# Создаём виджет для отображения фона
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Создаём элементы интерфейса
input_label = tk.Label(window, text="Введите слово(а):")
input_entry = tk.Entry(window, width=30)
encode_button = tk.Button(window, text="Закодировать", command=window_of_decoding, width=15)
decode_button = tk.Button(window, text="Декодировать", command=decode_button_click, width=15)

# Размещаем элементы в окне
input_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
input_entry.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
encode_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
decode_button.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

# Запускаем основной цикл обработки событий
window.mainloop()

