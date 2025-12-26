"""
Автотесты для поискового робота
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'crawler'))

from crawler import SearchCrawler


class TestCrawler(unittest.TestCase):
    """Тесты функциональности краулера"""
    
    @classmethod
    def setUpClass(cls):
        """Инициализация перед всеми тестами"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'crawler', 'config.yaml')
        cls.crawler = SearchCrawler(config_path)
    
    def test_01_config_loading(self):
        """Тест 1: Загрузка конфигурации"""
        self.assertIsNotNone(self.crawler.config)
        self.assertIn('db', self.crawler.config)
        self.assertIn('logic', self.crawler.config)
        print("Тест 1: Конфигурация загружена")
    
    def test_02_database_connection(self):
        """Тест 2: Подключение к БД"""
        self.assertIsNotNone(self.crawler.collection)
        # Проверяем, что можем выполнить запрос
        count = self.crawler.collection.count_documents({})
        self.assertGreaterEqual(count, 0)
        print(f"Тест 2: Подключение к БД (документов: {count})")
    
    def test_03_url_normalization(self):
        """Тест 3: Нормализация URL"""
        test_cases = [
            ('https://Habr.com/ru/articles/123/', 'https://habr.com/ru/articles/123'),
            ('https://lenta.ru/news/2025/12/20/test#comments', 'https://lenta.ru/news/2025/12/20/test'),
            ('HTTPS://EXAMPLE.COM/', 'https://example.com/'),
        ]
        
        for original, expected in test_cases:
            normalized = self.crawler.normalize_url(original)
            self.assertEqual(normalized, expected)
        
        print("Тест 3: Нормализация URL работает")
    
    def test_04_content_hash(self):
        """Тест 4: Вычисление хеша контента"""
        html1 = "<html><body>Test</body></html>"
        html2 = "<html><body>Test</body></html>"
        html3 = "<html><body>Different</body></html>"
        
        hash1 = self.crawler._calculate_content_hash(html1)
        hash2 = self.crawler._calculate_content_hash(html2)
        hash3 = self.crawler._calculate_content_hash(html3)
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        print("Тест 4: Хеширование контента работает")
    
    def test_05_queue_operations(self):
        """Тест 5: Операции с очередью"""
        initial_len = len(self.crawler.queue)
        
        self.crawler.add_to_queue('https://example.com/test', 'test_source')
        self.assertEqual(len(self.crawler.queue), initial_len + 1)
        
        # Проверяем, что дубликаты не добавляются
        self.crawler.visited_urls.add('https://example.com/duplicate')
        self.crawler.add_to_queue('https://example.com/duplicate', 'test_source')
        self.assertEqual(len(self.crawler.queue), initial_len + 1)
        
        print("Тест 5: Очередь работает корректно")
    
    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов"""
        cls.crawler.close()


if __name__ == '__main__':
    # Запуск тестов с выводом результатов
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCrawler)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Сохранение результатов в файл
    output_file = os.path.join(os.path.dirname(__file__), 'test_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Тестов запущено: {result.testsRun}\n")
        f.write(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Ошибок: {len(result.failures)}\n")
        f.write(f"Исключений: {len(result.errors)}\n")
    
    sys.exit(0 if result.wasSuccessful() else 1)