import os
import pandas as pd
cwd = os.getcwd()
os.chdir('./data/game_data')
game_folders = os.listdir()

# script to accumulate game data into single file
acc = []
for f in game_folders:
    if not f.startswith('.'):
      os.chdir(f) # move into directory
      # process file list
      csv_lst = os.listdir()
      for csv in csv_lst:
        df = pd.read_csv(csv, index_col=None, header=0)
        acc.append(df)
      # move back up to process next directory
      os.chdir('..')

super_df = pd.concat(acc, axis=0, ignore_index=True)
print(f'total_games in df {super_df.shape[0]}')
super_df.to_csv('all_games.csv')

