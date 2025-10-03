import pandas as pd
from tqdm import tqdm
from newspaper import Article
import os
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
        body = soup.find('div',{'class':'description feed-content'}).get_text()
        keywords = None
    elif 'larepublica' in url:
        body = ' '.join(x.get_text() for x in soup.find_all('p')[:-3])
        keywords = soup.find('meta',{'name':'keywords'}).get('content')
    elif 'elpopular' in url:
        body = ' '.join([x.get_text() for x in list(soup.find('div',{'class':'MainContent_main__body__LUkri'}).children) if x.name not in ['div','aside','style','script']])
        keywords = soup.find('meta',{'name':'keywords'}).get('content')
    else:
        raise NotImplemented(':/')

    return body, keywords, None

def extract_info_from_url(url, timeout = 25, user_agent = None):
    try:
        article = Article(url, language='es', request_timeout = timeout, browser_user_agent = user_agent) 
        article.download()
        article.parse()
        return article.text, article.meta_keywords, None  # No hay error
    except:
          try:
              return known_bad_sites(url, timeout=timeout, user_agent=user_agent)  # Intenta extraer información de sitios conocidos

          except Exception as e:
              return None, None, str(e)  # Si hay error, retorna el mensaje de error
    finally:
        time.sleep(random.uniform(0.3,1.7))

def cumulative_export(input_csv, output_csv, batch_size = 25, timeout = 25, user_agent = None, resume = True):
    buffer = []
    def flush_buffer():
        if not buffer:
            return
        out_df = pd.DataFrame(buffer, columns=["media_name", "publish_date", "title", "url", "body", "keywords", "error"])
        out_df.to_csv(
            output_csv,
            index=False,
            mode="a",
            header=not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0,
            encoding="utf-8",
        )
        buffer.clear()
    
    df_in = pd.read_csv(input_csv)

    processed = set()
    if resume and os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            processed = set(pd.read_csv(output_csv, usecols=["url"])["url"].astype(str))
        except Exception:
            pass

    for _, row in tqdm(df_in.iterrows(), total=len(df_in)):
        url = str(row['url'])
        if resume and url in processed:
            continue

        body, kws, err = extract_info_from_url(
            url,
            timeout=timeout,
            user_agent=user_agent
        )

        buffer.append({
            "media_name": row["media_name"],
            "publish_date": row["publish_date"],
            "title": row["title"],
            "url": url,
            "body": body,
            "keywords": kws,
            "error": err,
        })

        if len(buffer) >= batch_size:
            flush_buffer()

    flush_buffer()
    print(f'Done.')