import pandas as pd
from tqdm import tqdm
from newspaper import Article
import os
import nltk
import re
import requests as rq
import random
import time
import numpy as np
import glob
from waybackpy import WaybackMachineCDXServerAPI
nltk.download('punkt')

class scraper:
    def __init__(self, url, timeout, user_agent = None):
        self.timeout = timeout
        self.headers = {'user-agent': user_agent or 'Mozilla/5.0'}
        self.html, self.flag = self.autohandler(url)

    def vibe_check(self, url):
        try:
            time.sleep(random.uniform(0.3,1.7))

            GET = rq.get(url, headers = self.headers, timeout = self.timeout)
            HEADERS = GET.headers['Content-Type']
            MIMETYPE = HEADERS.split(';')[0]
            CHARSET = HEADERS.split('charset=')[-1]
            STATUS = GET.status_code

            if STATUS == 200 and 'text/html' in MIMETYPE:
                return GET.content.decode(CHARSET), True
            else:
                return None, False
        except:
            return None, False
            
    def autohandler(self, url):
        html, flag = self.vibe_check(url)

        if flag:
            return html, flag

        cdx = iter(
            WaybackMachineCDXServerAPI(
                url,
                user_agent = self.headers['user-agent']
            ).snapshots()
        )

        while True:
            try:
                snapshot = next(cdx)
            except StopIteration:
                break
            except Exception as e:
                continue

            try:
                html, flag = self.vibe_check(snapshot.archive_url)
                if flag:
                    return html, flag
            except:
                continue
        return None, False
        
    def extract_info(self):
        if not self.flag:
            return None, None, 'Unhandleable.'
        else:
            article = Article('',language='es')
            article.download(input_html = self.html)
            article.parse()
            return re.sub(r'\n\n',' ',article.text_cleaned), article.meta_keywords, None

class exporter:
    def __init__(
            self, 
            input_csv, output_csv, 
            batch_size = 25, timeout = 25, user_agent = None, resume = True, 
            distributed = True, total_workers = 8, wid = 0
        ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.batch_size = batch_size
        self.timeout = timeout
        self.user_agent = user_agent
        self.resume = resume
        self.distributed = distributed

        self.__buffer = []
        self.__init_df(total_workers, wid)

    def __worker_csv(self, wid: int) -> str:
        base, ext = os.path.splitext(self.output_csv)
        return f"{base}.w{wid:03d}{ext or '.csv'}"

    def __init_df(self, total_workers, wid):
        self.df_in = pd.read_csv(self.input_csv)
        self.processed = set()
        if self.resume and os.path.exists(self.output_csv) and os.path.getsize(self.output_csv) > 0:
            try:
                self.processed = set(pd.read_csv(self.output_csv, usecols=["url"])["url"].astype(str))
            except Exception:
                pass

        if self.distributed:
            assert total_workers > wid
            self.output_csv = self.__worker_csv(wid)
            self.df_in = self.df_in[~self.df_in['url'].isin(self.processed)]

            distrib = np.linspace(0,self.df_in.shape[0], total_workers + 1 , dtype = int)
            self.df_in = self.df_in.iloc[distrib[wid]:distrib[wid + 1]]
            self.processed = set()

    def __flush_buffer(self):
        if not self.__buffer:
            return
        out_df = pd.DataFrame(self.__buffer, columns=["media_name", "publish_date", "title", "url", "body", "keywords", "error"])
        out_df.to_csv(
            self.output_csv,
            index=False,
            mode="a",
            header=not os.path.exists(self.output_csv) or os.path.getsize(self.output_csv) == 0,
            encoding="utf-8",
        )
        self.__buffer.clear()

    def cumulative_export(self):
        for _, row in tqdm(self.df_in.iterrows(), total=len(self.df_in)):
            url = str(row['url'])
            if self.resume and url in self.processed:
                continue

            body, kws, err = scraper(
                url,
                timeout=self.timeout,
                user_agent=self.user_agent
            ).extract_info()

            self.__buffer.append({
                "media_name": row["media_name"],
                "publish_date": row["publish_date"],
                "title": row["title"],
                "url": url,
                "body": body,
                "keywords": kws,
                "error": err,
            })

            if len(self.__buffer) >= self.batch_size:
                self.__flush_buffer()

        self.__flush_buffer()

def merge_workers(dirwids, output_path):
    csvs = glob.glob('*.csv',root_dir=dirwids)
    dfs = pd.concat(
        [pd.read_csv(os.path.join(dirwids,f)) for f in csvs],
        ignore_index=True
    )
    dfs = dfs.drop_duplicates(subset=['url'], keep='last')
    dfs.to_csv(output_path, index=False)