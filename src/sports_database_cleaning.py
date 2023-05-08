import pandas as pd

# read the CSV file
df = pd.read_csv('../data/processing/betting_data/sports_database_2022.csv', header=None)

# add column names
df.columns = ['date', 'team_name', 'oppo_name', 'team_line', 'oppo_line', 'team_odds', 'oppo_odds']

# drop even number rows
df = df.iloc[1::2]

# change date format
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%m/%d/%Y')

# map team_name and oppo_name to team_id and oppo_id
mapping_1 = {
    'Hawks': '1610612737', 'Celtics': '1610612738', 'Pelicans': '1610612740', 'Bulls': '1610612741', 'Mavericks': '1610612742', 'Nuggets': '1610612743',
    'Rockets': '1610612745', 'Clippers': '1610612746', 'Lakers': '1610612747',
    'Heat': '1610612748', 'Bucks': '1610612749', 'Timberwolves': '1610612750', 'Nets': '1610612751', 'Knicks': '1610612752', 'Magic': '1610612753',
    'Pacers': '1610612754', 'Seventysixers': '1610612755', 'Suns': '1610612756',
    'Trailblazers': '1610612757', 'Kings': '1610612758', 'Spurs': '1610612759', 'Thunder': '1610612760', 'Raptors': '1610612761', 'Jazz': '1610612762',
    'Grizzlies': '1610612763', 'Wizards': '1610612764', 'Pistons': '1610612765', 'Hornets': '1610612766', 'Cavaliers': '1610612739', 'Warriors': '1610612744'
}

# Remove single quotes from team_id and oppo_id columns
df['team_id'] = df['team_name'].apply(lambda x: mapping_1[x.strip(" ").strip(("'"))])
df['oppo_id'] = df['oppo_name'].apply(lambda x: mapping_1[x.strip(" ").strip(("'"))])

# Drop team_name and oppo_name columns
df.drop(['team_name', 'oppo_name'], axis=1, inplace=True)

# save cleaned dataframe to CSV
df.to_csv('../data/processing/betting_data/cleaned_sports_database_2022.csv', index=False)
