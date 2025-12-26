#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <set>

#ifdef _WIN32
#include <windows.h>  // Для SetConsoleOutputCP
#endif

// Статистика токенизации
struct Statistics {
    size_t total_tokens = 0;
    size_t total_chars = 0;
    size_t unique_tokens = 0;
    double avg_token_length = 0.0;
    double time_seconds = 0.0;
    size_t input_size_bytes = 0;
};

// Чтение файла
std::string read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return "";
    }
    
    std::string content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    
    return content;
}

// Сохранение токенов
void save_tokens(const std::string& filename, const std::vector<std::string>& tokens) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Ошибка записи в файл: " << filename << std::endl;
        return;
    }
    
    // Записываем UTF-8 BOM для правильного чтения в Windows
    const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
    out.write(reinterpret_cast<const char*>(bom), sizeof(bom));
    
    for (const auto& token : tokens) {
        out << token << "\n";
    }
}

// Вычисление статистики
Statistics calculate_statistics(
    const std::vector<std::string>& tokens,
    double time_seconds,
    size_t input_size)
{
    Statistics stats;
    stats.total_tokens = tokens.size();
    stats.time_seconds = time_seconds;
    stats.input_size_bytes = input_size;
    
    // Подсчет символов и уникальных токенов
    std::set<std::string> unique_set(tokens.begin(), tokens.end());
    stats.unique_tokens = unique_set.size();
    
    for (const auto& token : tokens) {
        stats.total_chars += token.length();
    }
    
    if (stats.total_tokens > 0) {
        stats.avg_token_length = static_cast<double>(stats.total_chars) / stats.total_tokens;
    }
    
    return stats;
}

// Вывод статистики
void print_statistics(const Statistics& stats) {
    std::cout << "\n=== СТАТИСТИКА ТОКЕНИЗАЦИИ ===" << std::endl;
    std::cout << "Всего токенов: " << stats.total_tokens << std::endl;
    std::cout << "Уникальных токенов: " << stats.unique_tokens << std::endl;
    std::cout << "Средняя длина токена: " << std::fixed << std::setprecision(2) 
              << stats.avg_token_length << " символов" << std::endl;
    
    std::cout << "\n=== ПРОИЗВОДИТЕЛЬНОСТЬ ===" << std::endl;
    std::cout << "Размер входных данных: " << stats.input_size_bytes / 1024.0 << " КБ" << std::endl;
    std::cout << "Время обработки: " << std::fixed << std::setprecision(3) 
              << stats.time_seconds << " сек" << std::endl;
    
    if (stats.time_seconds > 0) {
        double speed_kb = (stats.input_size_bytes / 1024.0) / stats.time_seconds;
        double speed_tokens = stats.total_tokens / stats.time_seconds;
        
        std::cout << "Скорость: " << std::fixed << std::setprecision(2) 
                  << speed_kb << " КБ/сек" << std::endl;
        std::cout << "Скорость: " << std::fixed << std::setprecision(0) 
                  << speed_tokens << " токенов/сек" << std::endl;
    }
    std::cout << "==============================\n" << std::endl;
}

// Вывод справки
void print_help() {
    std::cout << "Использование: tokenizer [опции] <входной_файл> [выходной_файл]\n"
              << "\nОпции:\n"
              << "  -h, --help              Показать справку\n"
              << "  -l, --lowercase         Приводить к нижнему регистру (по умолчанию)\n"
              << "  -L, --no-lowercase      Не приводить к нижнему регистру\n"
              << "  -m, --min-length N      Минимальная длина токена (по умолчанию 1)\n"
              << "  -s, --stats             Показать статистику\n"
              << "\nПримеры:\n"
              << "  tokenizer input.txt output.txt\n"
              << "  tokenizer -s input.txt output.txt\n"
              << "  tokenizer -L -m 3 input.txt output.txt\n";
}

int main(int argc, char* argv[]) {
    // Установка UTF-8 для консоли
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif
    
    if (argc < 2) {
        print_help();
        return 1;
    }
    
    // Парсинг аргументов
    std::string input_file;
    std::string output_file;
    bool show_stats = false;
    bool lowercase = true;
    size_t min_length = 1;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        } else if (arg == "-s" || arg == "--stats") {
            show_stats = true;
        } else if (arg == "-l" || arg == "--lowercase") {
            lowercase = true;
        } else if (arg == "-L" || arg == "--no-lowercase") {
            lowercase = false;
        } else if (arg == "-m" || arg == "--min-length") {
            if (i + 1 < argc) {
                min_length = std::stoul(argv[++i]);
            }
        } else if (input_file.empty()) {
            input_file = arg;
        } else if (output_file.empty()) {
            output_file = arg;
        }
    }
    
    if (input_file.empty()) {
        std::cerr << "Ошибка: не указан входной файл" << std::endl;
        return 1;
    }
    
    // Если выходной файл не указан, выводим в stdout
    if (output_file.empty()) {
        output_file = "tokens.txt";
    }
    
    // Чтение входного файла
    std::cout << "Чтение файла: " << input_file << std::endl;
    std::string text = read_file(input_file);
    
    if (text.empty()) {
        std::cerr << "Ошибка: файл пуст или не найден" << std::endl;
        return 1;
    }
    
    // Настройка токенизатора
    Tokenizer tokenizer;
    tokenizer.set_lowercase(lowercase);
    tokenizer.set_min_token_length(min_length);
    
    // Токенизация с замером времени
    std::cout << "Токенизация..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> tokens = tokenizer.tokenize(text);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    // Сохранение результатов
    std::cout << "Сохранение токенов в: " << output_file << std::endl;
    save_tokens(output_file, tokens);
    
    // Статистика
    if (show_stats) {
        Statistics stats = calculate_statistics(tokens, duration.count(), text.size());
        print_statistics(stats);
    } else {
        std::cout << "Обработано токенов: " << tokens.size() << std::endl;
        std::cout << "Время: " << std::fixed << std::setprecision(3) 
                  << duration.count() << " сек" << std::endl;
    }
    
    return 0;
}