import pandas as pd
import numpy as np
from sklearn import *
import os, glob










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
WLoc = {'A': 1, 'H': 2, 'N': 3}
SecondaryTourney = {'NIT': 1, 'CBI': 2, 'CIT': 3, 'V16': 4, 'Regular': 5, 'NCAA': 6}

games = pd.concat((datafiles['NCAATourneyCompactResults'], datafiles['RegularSeasonCompactResults']), axis=0,
                  ignore_index=True)
games = pd.concat((games, datafiles['SecondaryTourneyCompactResults']), axis=0, ignore_index=True)
# games = pd.concat((datafiles['NCAATourneyDetailedResults'],datafiles['RegularSeasonDetailedResults']), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map(WLoc)
games['SecondaryTourney'] = games['SecondaryTourney'].map(SecondaryTourney)
games.head()

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

# Add Ranking
print("Add Ranking")
rank_reshaped_ad = np.load("../additional/rank_feat2.npy")
print(rank_reshaped_ad)
rank_reshaped_ad = pd.DataFrame(rank_reshaped_ad, columns=['1_', '2_','3_','4_','5_','6_','7_','8_','9_','10_',])
print(rank_reshaped_ad.head())
rank_reshaped = pd.read_csv("../additional/ranking_reshape.csv")
rank_reshaped = pd.concat([rank_reshaped[['Season', 'DayNum', 'Team']], rank_reshaped_ad], axis=1)
print(rank_reshaped.head())


print("rank_reshaped.shape", rank_reshaped.shape)
print(rank_reshaped.head())


rank_columns = rank_reshaped.columns
rank_columns1 = []
rank_columns2 = []
for column in rank_reshaped.columns:
    if column in ['Season', 'DayNum']:
        rank_columns1.append(column)
        rank_columns2.append(column)
    else:
        rank_columns1.append(column+'1')
        rank_columns2.append(column+'2')
rank_reshaped1 = pd.DataFrame(rank_reshaped, columns=rank_columns1)
rank_reshaped2 = pd.DataFrame(rank_reshaped, columns=rank_columns2)
games = pd.merge(games, rank_reshaped1, how='left', on=['Season', 'DayNum', 'Team1'])
games = pd.merge(games, rank_reshaped2, how='left', on=['Season', 'DayNum', 'Team2'])
# games = pd.merge(games, rank_reshaped2, on=['Season', 'DayNum', 'Team2'])
# games[rank_columns1] = games.fillna(games.mean())[rank_columns1]
# games[rank_reshaped2] = games[rank_reshaped2].fillna(games[rank_reshaped2].mean())
for column in rank_columns1 + rank_columns2:
    games[column] = games[column].fillna(games[column].mean() / games[column].max())


games = games.reset_index()
print(games.columns)
print(games.shape)


# Additional Features & Clean Up
print("Additional Features & Clean Up")
games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if r['Team1'] == r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)
# games = games.fillna(games.mean())
games.to_csv("../input/feature1.csv", index=None)

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
col = [c for c in x1.columns if
       c not in ['ID', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'Pred', 'DayNum', 'WTeamID', 'WScore',
                 'LTeamID', 'LScore', 'NumOT', 'ScoreDiff',
                 'Season',
                 'SecondaryTourney',
                 'ScoreDiffNorm',
                 'WLoc',
                 '',
                 '',


                 ]]
for season in sub['Season'].unique():
    print(season)
    x1 = games[((games['Season'] < int(season)) & (games['SecondaryTourney'] == 6))]
    # x1 = pd.concat((x1, games[((games['Season'] < int(int(season) + 1)) & (games['SecondaryTourney'] != 6))]), axis=0,
    #                ignore_index=True)
    x2 = games[((games['Season'] == int(season)) & (games['SecondaryTourney'] == 6))]
    test = sub[sub['Season'] == season]

    sdn = x1.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean()
    test = pd.merge(test, sdn, how='left', on=['IDTeams'])
    test['ScoreDiffNorm'] = test['ScoreDiffNorm'].fillna(0.)




    reg = linear_model.HuberRegressor()
    reg.fit(x1[col], x1['Pred'])
    pred = reg.predict(x2[col]).clip(0.05, 0.95)
    print('Log Loss:', metrics.log_loss(x2['Pred'], pred))
    # test['Pred'] = reg.predict(test[col])

    results.append(test)
results = pd.concat(results, axis=0, ignore_index=True).reset_index(drop=True)

# Testing for Sequence of Scoring
print("Testing for Sequence of Scoring")
results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
sub['Pred'] = sub['ID'].map(results).clip(0.05, 0.95).fillna(0.49)
# sub[['ID', 'Pred']].to_csv('../output/submission_rank3.csv', index=False)
"""
2014
Log Loss: 0.304945770188
2015
Log Loss: 0.331293014217
2016
Log Loss: 0.272694851265
2017
Log Loss: 0.319095848551

"""