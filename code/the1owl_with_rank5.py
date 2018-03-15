import pandas as pd
import numpy as np
from sklearn import *
import os, glob










datafiles = sorted(glob.glob('../input/**'))
# print(datafiles)
# datafiles = {file.split('/')[-1].split('.')[0]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
datafiles = {os.path.basename(file)[:-4]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
print(datafiles.keys())

# Add Seeds
print("Add Seeds")
seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in datafiles['NCAATourneySeeds'].values}
# Add 2018
if 2018 not in datafiles['NCAATourneySeeds']['Season'].unique():
    seeds = {**seeds, **{k.replace('2017_', '2018_'): seeds[k] for k in seeds if '2017_' in k}}


games = pd.read_csv("../input/feature1.csv")
print(games.head())
# Test Set
print("Test Set")
sub = datafiles['SampleSubmissionStage1']
sub['WLoc'] = 3  # N
sub['SecondaryTourney'] = 6  # NCAA
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

# Add Validation
print("Add Validation")
results = []
col = [c for c in games.columns if
       c not in ['ID', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'Pred', 'DayNum', 'WTeamID', 'WScore',
                 'LTeamID', 'LScore', 'NumOT', 'ScoreDiff',
    'ScoreDiffNorm',
    'index',
    'Team1Seed',
    'Team2Seed',
    'SeedDiff',


                 ]]
# col = ['SeedDiff']
print(col)

print("Add Validation")
results = []
loss = []
for season in sub['Season'].unique():
    print(season)
    x1 = games[((games['Season'] < int(season)) & (games['SecondaryTourney'] == 6))]
    # x1 = pd.concat((x1, games[((games['Season'] < int(int(season) + 1)) & (games['SecondaryTourney'] != 6))]), axis=0,
    #                ignore_index=True)
    x2 = games[((games['Season'] == int(season)) & (games['SecondaryTourney'] == 6))]
    test = sub[sub['Season'] == season]

    # sdn = x1.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean()
    # test = pd.merge(test, sdn, how='left', on=['IDTeams'])
    # test['ScoreDiffNorm'] = test['ScoreDiffNorm'].fillna(0.)
    #
    # # Interactions
    # inter = games[['IDTeam2', 'IDTeam1', 'Season', 'Pred']].rename(columns={'IDTeam2': 'Target', 'IDTeam1': 'Common'})
    # inter['Pred'] = inter['Pred'] * -1
    # inter = pd.concat((inter, games[['IDTeam1', 'IDTeam2', 'Season', 'Pred']].rename(
    #     columns={'IDTeam1': 'Target', 'IDTeam2': 'Common'})), axis=0, ignore_index=True).reset_index(drop=True)
    # inter = inter[((inter['Season'] <= int(season)) & (
    # inter['Season'] > int(season) - 2))]  # Only two years back and current regular season
    # inter = pd.merge(inter, inter, how='inner', on=['Common', 'Season'])
    # inter = inter[inter['Target_x'] != inter['Target_y']]
    # # inter['ID'] = inter.apply(lambda r: '_'.join(map(str, [r['Season']+1, r['Target_x'].split('_')[1],r['Target_y'].split('_')[1]])), axis=1)
    # inter['IDTeams'] = inter.apply(
    #     lambda r: '_'.join(map(str, [r['Target_x'].split('_')[1], r['Target_y'].split('_')[1]])), axis=1)
    # inter = inter[['IDTeams', 'Pred_x']]
    # inter = inter.groupby(['IDTeams'], as_index=False)[['Pred_x']].sum()
    # inter = {k: int(v) for k, v in inter.values}
    #
    # x1['Inter'] = x1['IDTeams'].map(inter).fillna(0)
    # x2['Inter'] = x2['IDTeams'].map(inter).fillna(0)
    # test['Inter'] = test['IDTeams'].map(inter).fillna(0)

    reg = linear_model.HuberRegressor()
    reg.fit(x1[col], x1['Pred'])
    pred = reg.predict(x2[col]).clip(0.05, 0.95)
    print('Log Loss:', metrics.log_loss(x2['Pred'], pred))
    loss.append(metrics.log_loss(x2['Pred'], pred))
    # test['Pred'] = reg.predict(test[col])

    results.append(test)
results = pd.concat(results, axis=0, ignore_index=True).reset_index(drop=True)

# Testing for Sequence of Scoring
print("Testing for Sequence of Scoring")
loss = np.mean(loss)
print('Total Log Loss:', loss)

results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
sub['Pred'] = sub['ID'].map(results).clip(0.05, 0.95).fillna(0.49)
# sub[['ID', 'Pred']].to_csv('../output/submission_rank3.csv', index=False)
"""
2014
Log Loss: 0.304952243832
2015
Log Loss: 0.331275987048
2016
Log Loss: 0.272698935641
2017
Log Loss: 0.31908357903
"""