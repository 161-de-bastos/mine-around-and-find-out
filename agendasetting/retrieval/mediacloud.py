from mediacloud.api import SearchApi
import dotenv
import os
import datetime as dt

dotenv.load_dotenv()

class mediacloud:
    def __init__(self, collection, start_date, end_date = dt.date.today()):
        self.__api = SearchApi(os.getenv('MC_API_KEY'))
        self.collection = collection
        self.start_date = start_date
        self.end_date = end_date
    
    def retrieve_stories(self, query):
        all_stories = []
        pag_token = None
        more_stories = True

        while more_stories:
            page, pag_token = self.__api.story_list(query, self.start_date, self.end_date, self.collection, pagination_token=pag_token)
            all_stories += page
            more_stories = pag_token is not None

        return all_stories

