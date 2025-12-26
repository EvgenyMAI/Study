import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class HabrParser:
    def __init__(self, output_dir='../corpus/habr', max_articles=20000):
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, 'raw')
        self.text_dir = os.path.join(output_dir, 'text')
        self.max_articles = max_articles
        self.ua = UserAgent()
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)
        
        self.session = requests.Session()
        
        # Список популярных хабов
        self.popular_hubs = [
            'python', 'javascript', 'web_development', 'programming', 
            'machine_learning', 'data_mining', 'artificial_intelligence',
            'algorithms', 'system_administration', 'information_security',
            'databases', 'dev_ops', 'mobile_development', 'game_development',
            'blockchain', 'cryptocurrency', 'neural_networks', 'cpp',
            'java', 'php', 'ruby', 'go_lang', 'rust', 'typescript',
            'frontend', 'backend', 'api', 'microservices', 'cloud_services',
            'aws', 'docker', 'kubernetes', 'git', 'linux', 'windows',
            'android_dev', 'ios_dev', 'reactjs', 'vuejs', 'angular',
            'nodejs', 'django', 'flask', 'spring', 'dotnet',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'data_engineering', 'big_data', 'deep_learning', 'nlp',
            'computer_vision', 'testing', 'qa', 'agile', 'scrum',
            'product_management', 'ui_ux', 'design', 'startup',
            'open_source', 'github', 'network_technologies', 'cybersecurity',
            'physics', 'mathematics', 'electronics', 'robotics',
            'internet_of_things', 'hardware', 'gadgets', 'diy'
        ]
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
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
    
    def get_article_urls(self):
        """Собирает URL статей через разные стратегии"""
        urls = self.load_existing_urls()
        initial_count = len(urls)
        
        print(f"Начинаем сбор ссылок (уже есть {initial_count} статей)...")
        
        # Стратегия 1: Основной раздел статей (насколько получится)
        print("\n[1/4] Собираем из основного раздела статей...")
        page = 1
        consecutive_errors = 0
        
        while len(urls) < self.max_articles and consecutive_errors < 3:
            try:
                url = f'https://habr.com/ru/articles/page{page}/'
                response = self.session.get(url, headers=self.get_headers(), timeout=10)
                
                if response.status_code != 200:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        print(f"Остановка на странице {page} после {consecutive_errors} ошибок")
                        break
                    time.sleep(2)
                    continue
                
                consecutive_errors = 0
                soup = BeautifulSoup(response.text, 'lxml')
                articles = soup.find_all('a', class_='tm-title__link')
                
                if not articles:
                    print(f"Нет статей на странице {page}")
                    break
                
                for article in articles:
                    href = article.get('href')
                    if href:
                        full_url = f"https://habr.com{href}" if href.startswith('/') else href
                        urls.add(full_url)
                
                if page % 20 == 0:
                    print(f"Страница {page}: всего {len(urls)} ссылок")
                
                page += 1
                time.sleep(0.4)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Ошибка на странице {page}: {e}")
                time.sleep(2)
                if consecutive_errors >= 3:
                    break
        
        print(f"Основной раздел: собрано {len(urls) - initial_count} новых ссылок")
        
        # Стратегия 2: ТОП статьи за разные периоды
        if len(urls) < self.max_articles:
            print("\n[2/4] Собираем ТОП статьи за разные периоды...")
            periods = ['alltime', 'yearly', 'monthly', 'weekly', 'daily']
            
            for period in periods:
                if len(urls) >= self.max_articles:
                    break
                
                print(f"  Период: {period}")
                page = 1
                max_pages = 200 if period == 'alltime' else 100
                
                while len(urls) < self.max_articles and page <= max_pages:
                    try:
                        url = f'https://habr.com/ru/articles/top/{period}/page{page}/'
                        response = self.session.get(url, headers=self.get_headers(), timeout=10)
                        
                        if response.status_code != 200:
                            break
                        
                        soup = BeautifulSoup(response.text, 'lxml')
                        articles = soup.find_all('a', class_='tm-title__link')
                        
                        if not articles:
                            break
                        
                        for article in articles:
                            href = article.get('href')
                            if href:
                                full_url = f"https://habr.com{href}" if href.startswith('/') else href
                                urls.add(full_url)
                        
                        page += 1
                        time.sleep(0.3)
                        
                    except Exception as e:
                        print(f"  Ошибка на странице {page} периода {period}: {e}")
                        break
                
                print(f"  {period}: всего {len(urls)} ссылок")
        
        # Стратегия 3: Популярные хабы
        if len(urls) < self.max_articles:
            print(f"\n[3/4] Собираем из {len(self.popular_hubs)} популярных хабов...")
            
            for hub_idx, hub in enumerate(self.popular_hubs):
                if len(urls) >= self.max_articles:
                    break
                
                page = 1
                max_pages = 100
                
                while len(urls) < self.max_articles and page <= max_pages:
                    try:
                        url = f'https://habr.com/ru/hubs/{hub}/articles/page{page}/'
                        response = self.session.get(url, headers=self.get_headers(), timeout=10)
                        
                        if response.status_code != 200:
                            break
                        
                        soup = BeautifulSoup(response.text, 'lxml')
                        articles = soup.find_all('a', class_='tm-title__link')
                        
                        if not articles:
                            break
                        
                        for article in articles:
                            href = article.get('href')
                            if href:
                                full_url = f"https://habr.com{href}" if href.startswith('/') else href
                                urls.add(full_url)
                        
                        page += 1
                        time.sleep(0.25)
                        
                    except Exception as e:
                        break
                
                if (hub_idx + 1) % 10 == 0:
                    print(f"Обработано {hub_idx + 1}/{len(self.popular_hubs)} хабов, всего {len(urls)} ссылок")
        
        # Стратегия 4: Посты и новости
        if len(urls) < self.max_articles:
            print("\n[4/4] Собираем посты и новости...")
            
            for section in ['news', 'posts']:
                if len(urls) >= self.max_articles:
                    break
                
                print(f"Раздел: {section}")
                page = 1
                
                while len(urls) < self.max_articles and page <= 100:
                    try:
                        url = f'https://habr.com/ru/{section}/page{page}/'
                        response = self.session.get(url, headers=self.get_headers(), timeout=10)
                        
                        if response.status_code != 200:
                            break
                        
                        soup = BeautifulSoup(response.text, 'lxml')
                        articles = soup.find_all('a', class_='tm-title__link')
                        
                        if not articles:
                            break
                        
                        for article in articles:
                            href = article.get('href')
                            if href:
                                full_url = f"https://habr.com{href}" if href.startswith('/') else href
                                urls.add(full_url)
                        
                        page += 1
                        time.sleep(0.3)
                        
                    except Exception as e:
                        break
                
                print(f"{section}: всего {len(urls)} ссылок")
        
        new_urls = [u for u in urls if u not in self.load_existing_urls()]
        print(f"\nИтого уникальных ссылок: {len(urls)} (новых: {len(new_urls)})")
        return list(urls)[:self.max_articles]
    
    def parse_article(self, url):
        """Парсит одну статью"""
        try:
            # Проверяем, не скачана ли уже
            article_id = re.search(r'/(\d+)/', url)
            article_id = article_id.group(1) if article_id else str(hash(url))
            
            text_file = os.path.join(self.text_dir, f'{article_id}.txt')
            if os.path.exists(text_file):
                return None  # Уже скачана
            
            response = self.session.get(url, headers=self.get_headers(), timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Извлекаем заголовок
            title_tag = soup.find('h1', class_='tm-title')
            title = title_tag.get_text(strip=True) if title_tag else "Без заголовка"
            
            # Извлекаем текст статьи
            article_body = soup.find('div', class_='tm-article-body')
            if not article_body:
                article_body = soup.find('div', class_='article-formatted-body')
            
            if not article_body:
                return None
            
            # Извлекаем метаданные
            author_tag = soup.find('a', class_='tm-user-info__username')
            author = author_tag.get_text(strip=True) if author_tag else "Неизвестен"
            
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
                f.write(f"Author: {author}\n")
                f.write(f"Date: {date}\n")
                f.write(f"URL: {url}\n")
                f.write(f"\n{text}\n")
            
            return {
                'id': article_id,
                'url': url,
                'title': title,
                'author': author,
                'date': date,
                'text_length': len(text),
                'raw_size': len(response.text)
            }
            
        except Exception as e:
            return None
    
    def download_all(self, num_workers=5):
        """Скачивает все статьи многопоточно"""
        urls = self.get_article_urls()
        
        # Фильтруем уже скачанные
        existing_urls = self.load_existing_urls()
        new_urls = [u for u in urls if u not in existing_urls]
        
        print(f"\nНачинаем скачивание {len(new_urls)} новых статей с Habr...")
        print(f"(Пропускаем {len(existing_urls)} уже скачанных)")
        
        results = []
        
        # Загружаем существующие метаданные
        metadata_file = os.path.join(self.output_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.parse_article, url): url for url in new_urls}
            
            for future in tqdm(as_completed(futures), total=len(new_urls), desc="Habr"):
                result = future.result()
                if result:
                    results.append(result)
                time.sleep(0.1)
        
        # Сохраняем метаданные
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nВсего статей с Habr: {len(results)} (добавлено новых: {len([r for r in results if r])})")
        return results

if __name__ == '__main__':
    parser = HabrParser(max_articles=20000)
    parser.download_all(num_workers=4)