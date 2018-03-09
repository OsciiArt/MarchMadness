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
rank = datafiles['MasseyOrdinals']
rank_types = sorted(rank['SystemName'].unique())
for i, rank_type in enumerate(rank_types):
    print("processeng: {}".format(rank_type))
    each_rank = rank[rank['SystemName']==rank_type]
    each_rank.columns = ['Season', 'DayNum', 'SystemName', 'Team', rank_type]
    if i==0:
        rank_reshaped = each_rank[['Season', 'DayNum', 'Team', rank_type]]
    else:
        rank_reshaped = pd.merge(rank_reshaped, each_rank[['Season', 'DayNum', 'Team', rank_type]],
                                 how='outer', on=['Season', 'DayNum', 'Team'])
rank_reshaped = rank_reshaped.fillna(rank_reshaped.mean())
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
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0., axis=1)
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
for season in sub['Season'].unique():
    print(season)
    x1 = games[((games['Season'] < int(season)) & (games['SecondaryTourney'] == 6))]
    x1 = pd.concat((x1, games[((games['Season'] < int(int(season) + 1)) & (games['SecondaryTourney'] != 6))]), axis=0,
                   ignore_index=True)
    x2 = games[((games['Season'] == int(season)) & (games['SecondaryTourney'] == 6))]
    test = sub[sub['Season'] == season]

    sdn = x1.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean()
    test = pd.merge(test, sdn, how='left', on=['IDTeams'])
    test['ScoreDiffNorm'] = test['ScoreDiffNorm'].fillna(0.)

    # Interactions
    inter = games[['IDTeam2', 'IDTeam1', 'Season', 'Pred']].rename(columns={'IDTeam2': 'Target', 'IDTeam1': 'Common'})
    inter['Pred'] = inter['Pred'] * -1
    inter = pd.concat((inter, games[['IDTeam1', 'IDTeam2', 'Season', 'Pred']].rename(
        columns={'IDTeam1': 'Target', 'IDTeam2': 'Common'})), axis=0, ignore_index=True).reset_index(drop=True)
    inter = inter[((inter['Season'] <= int(season)) & (
    inter['Season'] > int(season) - 2))]  # Only two years back and current regular season
    inter = pd.merge(inter, inter, how='inner', on=['Common', 'Season'])
    inter = inter[inter['Target_x'] != inter['Target_y']]
    # inter['ID'] = inter.apply(lambda r: '_'.join(map(str, [r['Season']+1, r['Target_x'].split('_')[1],r['Target_y'].split('_')[1]])), axis=1)
    inter['IDTeams'] = inter.apply(
        lambda r: '_'.join(map(str, [r['Target_x'].split('_')[1], r['Target_y'].split('_')[1]])), axis=1)
    inter = inter[['IDTeams', 'Pred_x']]
    inter = inter.groupby(['IDTeams'], as_index=False)[['Pred_x']].sum()
    inter = {k: int(v) for k, v in inter.values}

    x1['Inter'] = x1['IDTeams'].map(inter).fillna(0)
    x2['Inter'] = x2['IDTeams'].map(inter).fillna(0)
    test['Inter'] = test['IDTeams'].map(inter).fillna(0)
    col = [c for c in x1.columns if
           c not in ['ID', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'Pred', 'DayNum', 'WTeamID', 'WScore',
                     'LTeamID', 'LScore', 'NumOT', 'ScoreDiff']]

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
sub[['ID', 'Pred']].to_csv('../output/submission_rank.csv', index=False)
"""
mean
Log Loss: 0.314225603116
2015
Log Loss: 0.349731307348
2016
Log Loss: 0.290975087936
2017
Log Loss: 0.333975426288

maxscaler
nochange
"""