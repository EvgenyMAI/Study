import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import feedparser

class LentaParser:
    def __init__(self, output_dir='../corpus/lenta', max_articles=20000):
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, 'raw')
        self.text_dir = os.path.join(output_dir, 'text')
        self.max_articles = max_articles
        self.ua = UserAgent()
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
        
        self.session = requests.Session()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9',
            'Connection': 'keep-alive',
        }
    
    def load_existing_urls(self):
        """Загружает уже собранные URL из метаданных"""
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    existing = {item['url'] for item in metadata}
                    print(f"Найдено {len(existing)} уже скачанных статей")
                    return existing
            except:
                pass
        return set()
    
    def get_article_urls_from_rss(self):
        """Получает URLs из RSS ленты"""
        urls = set()
        rss_feeds = [
            'https://lenta.ru/rss/news',
            'https://lenta.ru/rss/articles',
            'https://lenta.ru/rss/last24'
        ]
        
        print("Собираем ссылки из RSS Lenta.ru...")
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    urls.add(entry.link)
                    if len(urls) >= self.max_articles:
                        break
                print(f"Из {feed_url}: {len(feed.entries)} ссылок")
                time.sleep(1)
            except Exception as e:
                print(f"Ошибка при парсинге RSS {feed_url}: {e}")
        
        return urls
    
    def get_article_urls_from_archive(self):
        """Собирает URLs из архива Лены за 4 года"""
        urls = self.load_existing_urls()
        initial_count = len(urls)
        
        print(f"\nСобираем ссылки из архива Lenta.ru (за 4 года)...")
        
        current_date = datetime.now()
        days_back = 0
        max_days = 1460  # 4 года
        
        while len(urls) < self.max_articles and days_back < max_days:
            try:
                date = current_date - timedelta(days=days_back)
                date_str = date.strftime('%Y/%m/%d')
                
                # URL архива за день
                archive_url = f'https://lenta.ru/{date_str}/'
                
                response = self.session.get(archive_url, headers=self.get_headers(), timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'lxml')
                    
                    # Ищем все ссылки на статьи
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        # Фильтруем ссылки на статьи (все типы контента)
                        if ('/news/' in href or '/articles/' in href or 
                            '/brief/' in href or '/story/' in href or 
                            '/photo/' in href or '/video/' in href):
                            full_url = f"https://lenta.ru{href}" if href.startswith('/') else href
                            # Убираем параметры из URL
                            full_url = full_url.split('?')[0]
                            urls.add(full_url)
                            
                            if len(urls) >= self.max_articles:
                                break
                
                if days_back % 30 == 0:
                    year = date.year
                    month = date.month
                    print(f"Дата {year}-{month:02d}: собрано {len(urls)} ссылок")
                
                days_back += 1
                time.sleep(0.2)
                
            except Exception as e:
                if days_back % 100 == 0:
                    print(f"Ошибка за {date_str}: {e}")
                days_back += 1
                time.sleep(0.5)
                continue
        
        new_count = len(urls) - initial_count
        print(f"\nИз архива собрано {len(urls)} уникальных ссылок (новых: {new_count})")
        return urls
    
    def parse_article(self, url):
        """Парсит одну статью Lenta.ru"""
        try:
            # Создаем ID из URL
            article_id = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            article_id = article_id.replace('.html', '')
            
            # Проверяем, не скачана ли уже
            text_file = os.path.join(self.text_dir, f'{article_id}.txt')
            if os.path.exists(text_file):
                return None
            
            response = self.session.get(url, headers=self.get_headers(), timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Извлекаем заголовок
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else "Без заголовка"
            
            # Извлекаем текст статьи (несколько вариантов селекторов)
            article_body = soup.find('div', class_='topic-body__content')
            if not article_body:
                article_body = soup.find('div', itemprop='articleBody')
            if not article_body:
                article_body = soup.find('div', class_='article__body')
            if not article_body:
                # Для кратких новостей
                article_body = soup.find('p', class_='topic-body__content-text')
            
            if not article_body:
                return None
            
            # Извлекаем дату
            date_tag = soup.find('time')
            date = date_tag.get('datetime') if date_tag else "Неизвестна"
            
            # Сохраняем сырой HTML
            raw_file = os.path.join(self.raw_dir, f'{article_id}.html')
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Извлекаем чистый текст
            text = article_body.get_text(separator='\n', strip=True)
            
            # Сохраняем текст и метаданные
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n")
                f.write(f"Date: {date}\n")
                f.write(f"URL: {url}\n")
                f.write(f"\n{text}\n")
            
            return {
                'id': article_id,
                'url': url,
                'title': title,
                'date': date,
                'text_length': len(text),
                'raw_size': len(response.text)
            }
            
        except Exception as e:
            return None
    
    def download_all(self, num_workers=3):
        """Скачивает все статьи"""
        # Сначала пробуем RSS
        urls = self.get_article_urls_from_rss()
        
        # Парсим архив за 4 года
        print(f"\nПарсим архив (нужно ~{self.max_articles} статей)...")
        archive_urls = self.get_article_urls_from_archive()
        urls.update(archive_urls)
        
        all_urls = list(urls)[:self.max_articles]
        
        # Фильтруем уже скачанные
        existing_urls = self.load_existing_urls()
        new_urls = [u for u in all_urls if u not in existing_urls]
        
        print(f"\nНачинаем скачивание {len(new_urls)} новых статей с Lenta.ru...")
        print(f"(Пропускаем {len(existing_urls)} уже скачанных)")
        
        results = []
        
        # Загружаем существующие метаданные
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.parse_article, url): url for url in new_urls}
            
            for future in tqdm(as_completed(futures), total=len(new_urls), desc="Lenta"):
                result = future.result()
                if result:
                    results.append(result)
                time.sleep(0.1)
        
        # Сохраняем метаданные
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nВсего статей с Lenta.ru: {len(results)}")
        return results

if __name__ == '__main__':
    parser = LentaParser(max_articles=20000)
    parser.download_all(num_workers=3)