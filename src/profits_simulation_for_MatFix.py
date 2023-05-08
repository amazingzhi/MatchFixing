import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime

# Load CSV file
file_path = "../data/processing/betting_data/merged_predictions_betting_dataset.csv"
df = pd.read_csv(file_path)

# Add team_odds_nor column
df['team_odds_nor'] = df.apply(lambda row: row['team_odds']/100 if row['team_odds'] >= 0 else -100/row['team_odds'], axis=1)

# Add T_or_F column
df['T_or_F'] = df.apply(lambda row: abs(row['pointspread'] - row['pointspread_pred']) <= abs(row['pointspread'] - (-row['team_line'])), axis=1)

# Calculate probability of True
p = df['T_or_F'].sum() / len(df)

# Add acc_sum column
acc_sum = [100]
b = 0.95
for i in range(1, len(df)):
    last_acc_sum = acc_sum[-1]
    if df.loc[i, 'T_or_F']:
        acc_sum.append(last_acc_sum + last_acc_sum * (p*b + p - 1))
    else:
        acc_sum.append(last_acc_sum * (1 - (p + (p - 1)/b)))
df['acc_sum'] = acc_sum

# Add accumulative return column
df['accumulative_return'] = (df['acc_sum'] - 100) / 100

# Plot graphs
fig, ax1 = plt.subplots()

ax1.plot(df['date'], df['acc_sum'], label='acc_sum', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('acc_sum', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(df['date'], df['accumulative_return'], label='accumulative_return', color='green')
ax2.set_ylabel('accumulative_return', color='green')
ax2.tick_params(axis='y', labelcolor='green')

ax1.xaxis.set_tick_params(rotation=45, labelsize=8)
ax1.set_xticks(df['date'][::30])
ax1.set_xticklabels(df['date'][::30])
plt.title('Accumulative Sum and Accumulative Return')
plt.show()
