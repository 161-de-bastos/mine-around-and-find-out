import pandas as pd
from tqdm import tqdm
from newspaper import Article
import nltk
import requests as rq
from bs4 import BeautifulSoup as bs
import random
import time
nltk.download('punkt')

def known_bad_sites(url, timeout, user_agent):
    headers = {'user-agent': user_agent or 'Mozilla/5.0'}
    resp = rq.get(url, headers=headers, timeout=timeout)
    soup = bs(resp.content.decode('utf-8'),'html.parser')

    if 'gob' in url:
        title = soup.find('h1').get_text()
        body = soup.find('div',{'class':'description feed-content'}).get_text()
        keywords = None
    elif 'larepublica' in url:
        title = soup.find('h1').get_text()
        body = ' '.join(x.get_text() for x in soup.find_all('p')[:-3])
        keywords = soup.find('meta',{'name':'keywords'}).get('content')
    elif 'elpopular' in url:
        title = soup.find('h1').get_text()
        body = ' '.join([x.get_text() for x in list(soup.find('div',{'class':'MainContent_main__body__LUkri'}).children) if x.name not in ['div','aside','style','script']])
        keywords = soup.find('meta',{'name':'keywords'}).get('content')
    else:
        raise NotImplemented(':/')

    return title, body, keywords, None

def extract_info_from_url(url, timeout = 25, user_agent = None):
    try:
        article = Article(url, language='es', request_timeout = timeout, browser_user_agent = user_agent) 
        article.download()
        article.parse()
        return article.title, article.text, article.meta_keywords, article.tags, None  # No hay error
    except:
          try:
              return known_bad_sites(url, timeout=timeout, user_agent=user_agent)  # Intenta extraer información de sitios conocidos

          except Exception as e:
              return None, None, None, None, str(e)  # Si hay error, retorna el mensaje de error
    finally:
        time.sleep(random.uniform(0.3,1.7))