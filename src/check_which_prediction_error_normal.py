# import packages
import os
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, norm, shapiro
import csv

# set working directory
os.chdir('../predictions/predictions_18_test_game/all_before_18/best_predictions_each_alg_train')

# parse csv files from directory
csv_files = [file for file in os.listdir() if file.endswith(".csv") and 'train' in file]
measures = {}
# loop through csv files
for file in csv_files:
    # read csv file
    temp_df = pd.read_csv(file)

    # calculate errors
    if 'DNN' in file:
        temp_df['pointspread_pred'] = temp_df['pointspread_pred'].str.strip('[]')
        temp_df['pointspread_pred'] = pd.to_numeric(temp_df['pointspread_pred'], errors='coerce')
        temp_df["PS_error"] = temp_df["pointspread"] - temp_df["pointspread_pred"]
        mean = temp_df["PS_error"].mean().round(3)
        std = temp_df["PS_error"].std().round(3)
        stat, p = kstest(temp_df["PS_error"], 't', args=(len(temp_df)-1,))
        stat, p = round(stat, 3), round(p, 3)
        measures[file] = [mean, std, stat, p]

        # check if values follow normal distribution
        sns.distplot(temp_df['PS_error'], fit=norm, kde=True)

    else:
        temp_df["PS_error"] = temp_df["pointspread"] - temp_df["pointspread_pred"]
        mean = temp_df["PS_error"].mean().round(3)
        std = temp_df["PS_error"].std().round(3)
        stat, p = kstest(temp_df["PS_error"], 't', args=(len(temp_df)-1,))
        stat, p = round(stat, 3), round(p, 3)
        # temp_df["PTS_error"] = temp_df["PTS_team"] - temp_df["PTS_team_pred"]
        # temp_df["PTS_to_PS_error"] = temp_df["pointspread"] - temp_df["pointspread_pred_from_pts"]
        # calculate mean and standard deviation
        # mean = list(temp_df[["PTS_error", "PTS_to_PS_error", "PS_error"]].mean().round(3))
        # std = list(temp_df[["PTS_error", "PTS_to_PS_error", "PS_error"]].std().round(3))
        # stat, p = shapiro(temp_df[["PTS_error", "PTS_to_PS_error", "PS_error"]])
        measures[file] = [mean, std, stat, p]

        # check if values follow normal distribution
        sns.distplot(temp_df['PS_error'], fit=norm, kde=True)
with open('../../../Result/master/measures.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, measures.keys())
    w.writeheader()
    w.writerow(measures)
print('finish')