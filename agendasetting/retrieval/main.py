EXPORTDIR = 'data'
VERSION = 1
TOTAL_WORKERS = 8
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

def extract_content(WID = 0):
    from agendasetting.retrieval.scraping import exporter, merge_workers
    from agendasetting.retrieval.config import SCRAPING

    expo = exporter(
        input_csv = f'data/retrieval_v{VERSION}.1.csv',
        output_csv = f'out/retrieval_v{VERSION}.2.csv',
        batch_size = 25,
        timeout = SCRAPING['timeout'],
        user_agent = SCRAPING['user_agent'],
        resume = True,
        distributed = True,
        total_workers = TOTAL_WORKERS,
        wid = WID  
    )
    expo.cumulative_export()
    merge_workers('out',f'data/retrieval_v{VERSION}.2.csv')

    print('Finished URL scraping.')

if __name__=='__main__':
    if not SKIP_SEQUENCE[0]:
        retrieve_urls()
    if not SKIP_SEQUENCE[1]:
        extract_content(WID = 0)
