EXPORTDIR = 'data'
VERSION = 1
SKIP_SEQUENCE = [
    False, 
    False
]

def retrieve_urls():
    from agendasetting.retrieval.mediacloud import mediacloud, build_query
    from agendasetting.retrieval.config import COLLECTIONS, MEDIACLOUD_START, ONTOPIC
    QUERY = build_query(ONTOPIC)
    client = mediacloud(VERSION,EXPORTDIR, COLLECTIONS, MEDIACLOUD_START, '2025-09-10')
    client.retrieve_stories(QUERY)

    print('Finished URL gathering.')

def extract_content():
    from agendasetting.retrieval.scraping import cumulative_export
    from agendasetting.retrieval.config import SCRAPING

    cumulative_export(
        input_csv = f'data/retrieval_v{VERSION}.1.csv',
        output_csv = f'data/retrieval_v{VERSION}.2.csv',
        batch_size = 25,
        timeout = SCRAPING['timeout'],
        user_agent = SCRAPING['user_agent'],
        resume = True
    )

    print('Finished URL scraping.')

if __name__=='__main__':
    if not SKIP_SEQUENCE[0]:
        retrieve_urls()
    if not SKIP_SEQUENCE[1]:
        extract_content()
