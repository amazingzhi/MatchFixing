import requests
import numpy as np
import json
from time import sleep
import pandas as pd
import os
import datetime

HEADERS = {'Referer': 'https://www.nba.com/stats/',
    'Origin': 'https://www.nba.com',
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Upgrade-Insecure-Requests': '1',}
USER_AGENTS = [
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    # 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0',
    # 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0'
]
HEADERS['User-Agent'] = np.random.choice(USER_AGENTS)

def get_date(days: int):
    date = datetime.date.today() - datetime.timedelta(days=days)
    return date.strftime('%m/%d/%Y')


# Game by day
dataset_to_keep = ['GameHeader','LineScore','LastMeeting','EastConfStandingsByDay','WestConfStandingsByDay']
dfs = {}
wait = True

for i in range(1,10000):
    date = get_date(i)
    url = 'https://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate='+date

    if date == '09/30/2020':
        break

    response = requests.get(url, headers=HEADERS)
    status_200_ok = response.status_code == 200
    nb_error = 0

    while not status_200_ok and nb_error < 5:
        print(response.status_code, url)
        sleep(1)
        response = requests.get(url)
        status_200_ok = response.status_code == 200
        nb_error += 1

    print(response.status_code, url)

    if nb_error < 5:
        nba_day_json = response.json()

        for dataset in nba_day_json['resultSets']:
            df_name = dataset['name']
            df_head = dataset['headers']
            df_rows = dataset['rowSet']
            if df_name not in dataset_to_keep:
                continue

            new_df = pd.DataFrame(df_rows, columns=df_head)
            if df_name not in dfs:
                dfs[df_name] = new_df
            else: 
                dfs[df_name] = pd.concat([dfs[df_name], new_df])


for name in dfs:
    path = 'data/'+str(name)+'.csv'
    if os.path.isfile(path):
        print('exist')
    dfs[name].to_csv(path, index=False)