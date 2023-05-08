import pandas as pd

# read in the prediction and betting datasets
prediction = pd.read_csv(
    '../predictions/predictions_20_22_test_game/all_before_20/feature_selection_LR_standard_prediction.csv')
betting = pd.read_csv('../data/processing/betting_data/cleaned_sports_database_2022.csv')

# Convert date columns to datetime format
prediction['GAME_DATE_EST'] = pd.to_datetime(prediction['GAME_DATE_EST'], format='%m/%d/%Y')
betting['date'] = pd.to_datetime(betting['date'], format='%m/%d/%Y')

# rename the columns in the prediction dataset
prediction = prediction.rename(columns={'GAME_DATE_EST': 'date'})
betting['team_id'] = betting['team_id'].astype(int)
betting['oppo_id'] = betting['oppo_id'].astype(int)
# merge the two datasets based on the common columns
merged = pd.merge(betting, prediction, on=['date', 'team_id', 'oppo_id'], how='left')

# check for missing data and multiple matches
missing_data = merged[merged.isnull().any(axis=1)]
if len(missing_data) > 0:
    print('The following rows in the betting dataset could not be found in the prediction dataset:')
    print(missing_data)

multiple_matches = merged[merged.duplicated(['date', 'team_id', 'oppo_id'], keep=False)]
if len(multiple_matches) > 0:
    print('The following rows in the betting dataset have multiple matches in the prediction dataset:')
    print(multiple_matches)

# save the merged dataset to a new CSV file
merged.to_csv('../data/processing/betting_data/merged_predictions_betting_dataset.csv', index=False)
