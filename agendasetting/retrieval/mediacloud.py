from mediacloud.api import SearchApi
import dotenv
import os
import datetime as dt
import pandas as pd
import time
dotenv.load_dotenv()

class mediacloud:
    def __init__(self, exportdir, collection, start_date, end_date = dt.date.today()):
        self.__api = SearchApi(os.getenv('API_KEY'))
        self.exportdir = exportdir
        self.collection = collection
        self.start_date = dt.date.fromisoformat(start_date)
        self.end_date = dt.date.fromisoformat(end_date) if type(end_date) == str else end_date
    
    def hit_export(self, jsons):
        df = pd.DataFrame(jsons)[['media_name', 'publish_date','title','url']]
        realpath = os.path.join(self.exportdir, 'retrieved.csv')
        df.to_csv(
            realpath, 
            index = False,
            mode = 'a',
            header = not os.path.exists(realpath)
        )

    def retrieve_stories(self, query, pag_token = None):
        while True:
            page, pag_token = self.__api.story_list(
                query, 
                self.start_date, 
                self.end_date, 
                self.collection, 
                pagination_token = pag_token
            )
            self.hit_export(page)
            del page

            if not pag_token:
                break
            else:
                with open(os.path.join(self.exportdir,'tokenlog.txt'),'w') as f:
                    f.write(str(pag_token))

            time.sleep(30) # Two calls per minute rate limit
    
def build_query(terms):
    parts = [f'"{t}"' for t in terms]
    return " OR ".join(parts)
