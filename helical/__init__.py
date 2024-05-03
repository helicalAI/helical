import os


CACHE_DIR_HELICAL= '~/.cache/helical/model'
if not os.path.exists(CACHE_DIR_HELICAL):
    os.makedirs(CACHE_DIR_HELICAL)