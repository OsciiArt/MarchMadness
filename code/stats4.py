import pandas as pd
import numpy as np
from sklearn import *
import os, glob
from sklearn.linear_model import LogisticRegression

datafiles = sorted(glob.glob('../input/**'))
# print(datafiles)
# datafiles = {file.split('/')[-1].split('.')[0]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
datafiles = {os.path.basename(file)[:-4]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
print(datafiles.keys())

datafiles['NCAATourneyCompactResults']['SecondaryTourney'] = 'NCAA'
datafiles['NCAATourneyDetailedResults']['SecondaryTourney'] = 'NCAA'
datafiles['RegularSeasonCompactResults']['SecondaryTourney'] = 'Regular'
datafiles['RegularSeasonDetailedResults']['SecondaryTourney'] = 'Regular'

# Presets
print("Presets")
games = datafiles['NCAATourneyCompactResults']

# Add Ids
print("Add Ids")
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# inverse
games2 = games.copy()
games2['ID'] = games['ID'].apply(lambda r: r[:4] + r[9:14] + r[4:9])
games2['IDTeams'] = games['IDTeams'].apply(lambda r: r[5:9] + "_" + r[:4])
games2['Team1'] = games['Team2']
games2['Team2'] = games['Team1']
games2['IDTeam1'] = games['IDTeam2']
games2['IDTeam2'] = games['IDTeam1']
games = pd.concat([games, games2])

# Add Seeds
print("Add Seeds")
seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in datafiles['NCAATourneySeeds'].values}
# Add 2018
if 2018 not in datafiles['NCAATourneySeeds']['Season'].unique():
    seeds = {**seeds, **{k.replace('2017_', '2018_'): seeds[k] for k in seeds if '2017_' in k}}

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)


print(games.columns)
print(games.shape)


# Additional Features & Clean Up
print("Additional Features & Clean Up")
games['Pred'] = games.apply(lambda r: 1. if r['Team1'] == r['WTeamID'] else 0., axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

games = pd.read_csv("../additional/df_predictions_tourney.csv")
print(games.columns)
"""
['Seed', 'OrdinalRank', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT',
       'FT_RATE', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO',
       'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience', 'result', 'Season',
       'WTeamID', 'LTeamID'],

"""
# games = games.fillna(games.mean())

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
# sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

# Add Validation
print("Add Validation")
results = []
loss = []
col = ['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
print(col)
loss = []
# print(col_i)
for season in sub['Season'].unique():
    print(season)
    x1 = games.loc[((games['Season'] < int(season)))]
    x2 = games.loc[((games['Season'] == int(season)))]
    test = sub[sub['Season'] == season]
    reg = linear_model.HuberRegressor()
    reg2 = LogisticRegression()
    reg.fit(x1[col], x1['result'])
    reg2.fit(x1[col], x1['result'])
    pred = reg.predict(x2[col]).clip(0.05, 0.95)
    reg = (0.5 * reg + 0.5 * reg2) /2
    # pred[np.where((pred>=0.6) & (pred<0.7))] = 0.7
    # pred[np.where((pred>=0.3) & (pred<0.4))] = 0.3
    print('Log Loss:', metrics.log_loss(x2['result'], pred))
    loss.append(metrics.log_loss(x2['result'], pred))
loss = np.mean(loss)
print('Total Log Loss:', loss)
    # test['Pred'] = reg.predict(test[col])
#
#     results.append(test)
# results = pd.concat(results, axis=0, ignore_index=True).reset_index(drop=True)

# Testing for Sequence of Scoring
print("Testing for Sequence of Scoring")
loss = np.mean(loss)
print('Total Log Loss:', loss)

results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
sub['Pred'] = sub['ID'].map(results).clip(0.05, 0.95).fillna(0.49)
"""
pred[np.where((pred>=0.6) & (pred<0.7))] = 0.7
pred[np.where((pred>=0.3) & (pred<0.4))] = 0.3
2014
Log Loss: 0.592100625065
2015
Log Loss: 0.490801488816
2016
Log Loss: 0.581148919285
2017
Log Loss: 0.526696621486
Total Log Loss: 0.547686913663
Testing for Sequence of Scoring
Total Log Loss: 0.547686913663

"""