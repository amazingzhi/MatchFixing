import csv
from configparser import ConfigParser
import numpy as np
import copy
import pandas as pd

# global variables
file = '../configs/game_time_series_data_creation.ini'
config = ConfigParser()
config.read(file)
data_path = config['paths']['read_path']
out_path = config['paths']['save_path']

def find_all_games_by_given_team_id(team_id, df):
    df_home = df[df['HOME_TEAM_ID'] == team_id]
    df_away = df[df['VISITOR_TEAM_ID'] == team_id]
    df_team = pd.concat([df_home,df_away], ignore_index=True)
    return df_team

def create_new_df(df, team_id):
    new_cols = ['GAME_DATE_EST', 'GAME_ID', 'team_id', 'oppo_id', 'year', 'month', 'day', 'pointspread', 'loc',
                'PTS_team',	'FG_PCT_team',	'FT_PCT_team',	'FG3_PCT_team',	'AST_team',	'REB_team', 'FGM_team',
                'FGA_team',	'FG3M_team',	'FG3A_team',	'FTM_team',	'FTA_team',	'OREB_team',	'DREB_team',
                'STL_team',	'BLK_team',	'TO_team',	'PF_team',	'PLUS_MINUS_team', 'EFG%_team', 'PPS_team', 'FIC_team',
                'PTS_oppo', 'FG_PCT_oppo', 'FT_PCT_oppo', 'FG3_PCT_oppo', 'AST_oppo', 'REB_oppo', 'FGM_oppo',
                'FGA_oppo', 'FG3M_oppo', 'FG3A_oppo', 'FTM_oppo', 'FTA_oppo', 'OREB_oppo', 'DREB_oppo',
                'STL_oppo', 'BLK_oppo', 'TO_oppo', 'PF_oppo', 'PLUS_MINUS_oppo', 'EFG%_oppo', 'PPS_oppo', 'FIC_oppo'
                ]
    list_data = df.values.tolist()
    new_games = []
    for row in list_data:
        if row[3] == team_id:
            new_games.append(row[:2] + row[3:5] + row[-3:] + [row[-4]] + [1]
                             + row[7:13] + row[21:34] + [row[-10], row[-8], row[-6]]
                             + row[14:20] + row[34:47] + [row[-9], row[-7], row[-5]])
        elif row[4] == team_id:
            new_games.append(row[:2] + row[4:2:-1] + row[-3:] + [-row[-4]] + [0]
                             + row[14:20] + row[34:47] + [row[-9], row[-7], row[-5]]
                             + row[7:13] + row[21:34] + [row[-10], row[-8], row[-6]])
        else:
            print(f'error {row}')
    new_df = pd.DataFrame(new_games, columns=new_cols)
    return new_df


def main():
    df = pd.read_csv(data_path)
    team_ids = df.HOME_TEAM_ID.unique()
    # add year month day
    df['year'], df['month'], df['day'] = df.apply(lambda row: int(row['GAME_DATE_EST'].split('/')[2]), axis=1),\
                                         df.apply(lambda row: int(row['GAME_DATE_EST'].split('/')[0]), axis=1),\
                                         df.apply(lambda row: int(row['GAME_DATE_EST'].split('/')[1]), axis=1)
    for i, team_id in enumerate(team_ids):
        df_team = find_all_games_by_given_team_id(team_id, df)
        df_team_sorted = df_team.sort_values(by=['year', 'month', 'day'], ascending=[True, True, True])
        df_new_team = create_new_df(df_team_sorted, team_id)
        if i == 0:
            df_new_teams = df_new_team
        else:
            df_new_teams = pd.concat([df_new_teams, df_new_team], ignore_index=True)
    df_new_teams.to_csv(out_path, index=False)



if __name__ == '__main__':
    main()
