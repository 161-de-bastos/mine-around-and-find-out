from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import time
import datetime
from tqdm import tqdm
import os

options = Options()
options.add_argument('--headless')
options.set_preference('permissions.default.image',2)

def google_crawl(name,filepath):
    def get_source(url):
        driver = webdriver.Firefox(options=options)
        driver.implicitly_wait(10)
        driver.get(url)

        time.sleep(3)
        soup = bs(driver.page_source, 'html.parser')
        driver.close()

        return soup

    def google_url(prompt):
        '''
        search?q    : query de consulta para el buscador. Para ser aceptados, los espacios deben ser "+".
        tbm=nws     : sección de noticias.
        tbs=sbd:1   : ordenamiento de resultados por fecha.
        lr=lang_es  : resultados en lenguaje español
        cr=countryPE: resultados de Perú
        '''

        full = prompt.replace(' ','+')
        return f'https://www.google.com/search?q={full}&tbm=nws&tbs=sbd:1&lr=lang_es&cr=countryPE&hl=es'

    def crawler(url):
        i = 2
        collection = {
            'mine':[],
            'link':[],
            'date':[]
        }

        try:
            while True:
                latestSoup = get_source(url)
                time.sleep(3)

                # Si el enlace provisto genera resultados.
                if latestSoup.find_all('a',{'class':'WlydOe'}):
                    for page in latestSoup.find_all('a',{'class':'WlydOe'}):
                        collection['mine'] += [name]
                        collection['link'] += [page.get('href')]
                        collection['date'] += [page.find_all('span')[-1].text]

                    next_page = latestSoup.find('a',{'aria-label':f'Page {i}'})

                    # Si existe una página luego del parseado.
                    if next_page:
                        url = next_page.get('href')
                        i += 1
                    else:
                        break
                else:
                    break
        except Exception as e:
            print(f'Error: {e}')

        return pd.DataFrame(collection)

    dates = [
        datetime.datetime.fromisocalendar(y,w,1).strftime('%Y-%m-%d') for y in range(2005,2025) for w in range(1,53)
    ]

    for i in tqdm(range(len(dates)-1)):
        date = f' after:{dates[i]} before:{dates[i+1]}'
        df = crawler(google_url(name + date))
        df.to_csv(f'{filepath}/data/{name}_urls.csv',index=False, mode='a', header=not os.path.exists(f'{filepath}/data/{name}_urls.csv'))