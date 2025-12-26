"""
Поисковый робот для ЛР2
Скачивает документы и сохраняет в MongoDB
"""

import sys
import yaml
import time
import logging
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin, urlunparse
from typing import Dict, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError


class SearchCrawler:
    """Поисковый робот с поддержкой остановки/возобновления и переобкачки"""
    
    def __init__(self, config_path: str):
        """Инициализация робота из YAML-конфига"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_database()
        self._setup_session()
        
        self.visited_urls: Set[str] = set()
        self.queue: List[Dict] = []
        
        logging.info("Поисковый робот инициализирован")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации из YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Настройка логирования"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'crawler.log')
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _setup_database(self):
        """Подключение к MongoDB и создание индексов"""
        db_config = self.config['db']
        
        try:
            self.client = MongoClient(
                db_config['host'],
                db_config['port'],
                serverSelectionTimeoutMS=5000
            )
            # Проверка подключения
            self.client.server_info()
            
            self.db = self.client[db_config['database']]
            self.collection = self.db[db_config['collection']]
            
            # Создаем индексы для оптимизации
            self.collection.create_index([('url', ASCENDING)], unique=True)
            self.collection.create_index([('source', ASCENDING)])
            self.collection.create_index([('crawled_at', DESCENDING)])
            self.collection.create_index([('content_hash', ASCENDING)])
            
            # Коллекция для очереди URL
            self.queue_collection = self.db['crawl_queue']
            self.queue_collection.create_index([('priority', DESCENDING)])
            
            logging.info(f"Подключено к MongoDB: {db_config['database']}")
            
        except Exception as e:
            logging.error(f"Ошибка подключения к MongoDB: {e}")
            sys.exit(1)
    
    def _setup_session(self):
        """Настройка HTTP-сессии"""
        self.session = requests.Session()
        self.ua = UserAgent()
    
    def _get_headers(self) -> Dict[str, str]:
        """Генерация HTTP-заголовков"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def normalize_url(self, url: str) -> str:
        """Нормализация URL (удаление фрагментов, параметров сессии)"""
        parsed = urlparse(url)
        
        # Удаляем фрагмент (#...)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # fragment удаляем
        ))
        
        # Удаляем trailing slash если это не корень
        if normalized.endswith('/') and normalized.count('/') > 3:
            normalized = normalized[:-1]
        
        return normalized
    
    def _calculate_content_hash(self, html: str) -> str:
        """Вычисление хеша содержимого для обнаружения изменений"""
        return hashlib.md5(html.encode('utf-8')).hexdigest()
    
    def _should_recrawl(self, doc: Dict) -> bool:
        """Проверка, нужно ли переобкачать документ"""
        recrawl_period = self.config['logic'].get('recrawl_period_days', 30)
        crawled_at = doc.get('crawled_at', 0)
        
        # Проверяем, прошло ли достаточно времени
        time_diff = time.time() - crawled_at
        days_diff = time_diff / (24 * 3600)
        
        return days_diff >= recrawl_period
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Скачивание HTML-страницы"""
        logic_config = self.config['logic']
        timeout = logic_config.get('request_timeout', 10)
        retry_attempts = logic_config.get('retry_attempts', 3)
        
        for attempt in range(retry_attempts):
            try:
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response.text
                else:
                    logging.warning(f"HTTP {response.status_code} для {url}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Попытка {attempt + 1}/{retry_attempts} для {url}: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка
        
        logging.error(f"Не удалось скачать {url} после {retry_attempts} попыток")
        return None
    
    def save_document(self, url: str, html: str, source: str) -> bool:
        """Сохранение документа в MongoDB"""
        normalized_url = self.normalize_url(url)
        content_hash = self._calculate_content_hash(html)
        crawled_at = int(time.time())
        
        document = {
            'url': normalized_url,
            'html': html,
            'source': source,
            'crawled_at': crawled_at,
            'content_hash': content_hash,
            'size': len(html)
        }
        
        try:
            # Проверяем, существует ли документ
            existing = self.collection.find_one({'url': normalized_url})
            
            if existing:
                # Проверяем, изменился ли контент
                if existing.get('content_hash') != content_hash:
                    # Обновляем документ
                    self.collection.update_one(
                        {'url': normalized_url},
                        {'$set': document}
                    )
                    logging.info(f"Обновлен (изменился): {normalized_url}")
                    return True
                else:
                    # Контент не изменился, обновляем только дату
                    self.collection.update_one(
                        {'url': normalized_url},
                        {'$set': {'crawled_at': crawled_at}}
                    )
                    logging.info(f"○ Не изменился: {normalized_url}")
                    return False
            else:
                # Новый документ
                self.collection.insert_one(document)
                logging.info(f"Добавлен: {normalized_url}")
                return True
                
        except DuplicateKeyError:
            logging.warning(f"Дубликат URL: {normalized_url}")
            return False
        except Exception as e:
            logging.error(f"Ошибка сохранения {normalized_url}: {e}")
            return False
    
    def load_queue_from_db(self):
        """Загрузка очереди из БД (для возобновления после остановки)"""
        queue_items = self.queue_collection.find().sort('priority', DESCENDING)
        self.queue = list(queue_items)
        
        if self.queue:
            logging.info(f"Загружено {len(self.queue)} URL из очереди")
        else:
            logging.info("Очередь пуста, начинаем с начальных URL")
    
    def save_queue_to_db(self):
        """Сохранение очереди в БД (для возможности остановки)"""
        if not self.queue:
            return
        
        # Очищаем старую очередь
        self.queue_collection.delete_many({})
        
        # Сохраняем текущую
        if self.queue:
            self.queue_collection.insert_many(self.queue)
        
        logging.info(f"Сохранено {len(self.queue)} URL в очередь")
    
    def add_to_queue(self, url: str, source: str, priority: int = 0):
        """Добавление URL в очередь"""
        normalized = self.normalize_url(url)
        
        if normalized not in self.visited_urls:
            self.queue.append({
                'url': normalized,
                'source': source,
                'priority': priority
            })
    
    def extract_links(self, base_url: str, html: str, source: str) -> List[str]:
        """Извлечение ссылок со страницы"""
        links = []
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Преобразуем относительные ссылки в абсолютные
                full_url = urljoin(base_url, href)
                normalized = self.normalize_url(full_url)
                
                # Фильтруем только нужные URL
                parsed = urlparse(normalized)
                if source == 'habr' and 'habr.com' in parsed.netloc:
                    if '/articles/' in parsed.path or '/news/' in parsed.path:
                        links.append(normalized)
                
                elif source == 'lenta' and 'lenta.ru' in parsed.netloc:
                    if '/news/' in parsed.path or '/articles/' in parsed.path:
                        links.append(normalized)
            
        except Exception as e:
            logging.error(f"Ошибка извлечения ссылок из {base_url}: {e}")
        
        return links
    
    def crawl(self):
        """Основной процесс обкачки"""
        logic_config = self.config['logic']
        delay = logic_config.get('delay_between_requests', 0.5)
        max_docs = logic_config.get('max_documents_per_run', 100)
        
        # Загружаем очередь из БД (возобновление)
        self.load_queue_from_db()
        
        # Если очередь пуста, заполняем начальными URL
        if not self.queue:
            for source_config in self.config['sources']:
                source_name = source_config['name']
                for url in source_config['start_urls']:
                    self.add_to_queue(url, source_name, priority=10)
        
        processed = 0
        new_docs = 0
        updated_docs = 0
        
        logging.info(f"Начинаем обкачку (макс. {max_docs} документов)")
        
        try:
            while self.queue and processed < max_docs:
                # Берем URL из очереди
                item = self.queue.pop(0)
                url = item['url']
                source = item['source']
                
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                
                # Проверяем, нужно ли переобкачать
                existing = self.collection.find_one({'url': url})
                if existing and not self._should_recrawl(existing):
                    logging.info(f"Пропускаем (свежий): {url}")
                    continue
                
                # Скачиваем страницу
                html = self.fetch_page(url)
                
                if html:
                    # Сохраняем документ
                    is_new = self.save_document(url, html, source)
                    
                    if is_new:
                        if existing:
                            updated_docs += 1
                        else:
                            new_docs += 1
                    
                    # Извлекаем ссылки для дальнейшей обкачки
                    links = self.extract_links(url, html, source)
                    for link in links:
                        self.add_to_queue(link, source, priority=5)
                    
                    processed += 1
                
                # Задержка между запросами
                time.sleep(delay)
                
                # Периодически сохраняем очередь
                if processed % 10 == 0:
                    self.save_queue_to_db()
        
        except KeyboardInterrupt:
            logging.info("\nПолучен сигнал остановки (Ctrl+C)")
        
        finally:
            # Сохраняем очередь перед выходом
            self.save_queue_to_db()
            
            # Статистика
            total_in_db = self.collection.count_documents({})
            
            logging.info("=" * 60)
            logging.info("СТАТИСТИКА ОБКАЧКИ:")
            logging.info(f"Обработано страниц: {processed}")
            logging.info(f"Новых документов: {new_docs}")
            logging.info(f"Обновлено документов: {updated_docs}")
            logging.info(f"В очереди осталось: {len(self.queue)}")
            logging.info(f"Всего в БД: {total_in_db}")
            logging.info("=" * 60)
    
    def get_statistics(self) -> Dict:
        """Получение статистики из БД"""
        stats = {
            'total_documents': self.collection.count_documents({}),
            'by_source': {},
            'queue_size': self.queue_collection.count_documents({})
        }
        
        for source_config in self.config['sources']:
            source_name = source_config['name']
            count = self.collection.count_documents({'source': source_name})
            stats['by_source'][source_name] = count
        
        return stats
    
    def close(self):
        """Закрытие подключений"""
        if hasattr(self, 'client'):
            self.client.close()
            logging.info("Подключение к MongoDB закрыто")


def main():
    """Точка входа"""
    if len(sys.argv) != 2:
        print("Использование: python crawler.py <путь_к_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    crawler = SearchCrawler(config_path)
    
    try:
        crawler.crawl()
        
        # Выводим статистику
        stats = crawler.get_statistics()
        print("\nИтоговая статистика:")
        print(f"Всего документов: {stats['total_documents']}")
        for source, count in stats['by_source'].items():
            print(f"{source}: {count}")
        print(f"В очереди: {stats['queue_size']}")
        
    finally:
        crawler.close()


if __name__ == '__main__':
    main()