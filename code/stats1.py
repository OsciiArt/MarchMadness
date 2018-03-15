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
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

# Add Validation
print("Add Validation")
results = []
loss = []
col = [
    'Seed',
    # 'OrdinalRank',
    'PIE',
    # 'FG_PCT',
    # 'TURNOVER_RATE',
    # 'OFF_REB_PCT',
    'FT_RATE',
    # '4FACTOR',
    # 'OFF_EFF',
    # 'DEF_EFF',
    # 'ASSIST_RATIO',
    # 'DEF_REB_PCT',
    # 'FT_PCT',
    # 'WINPCT',
    # 'experience',
    # 'result',
    # 'Season',
    # 'WTeamID',
    # 'LTeamID',
]
print(col)
for i in range(len(col)):
    col_i = col[:i] + col[i+1:]
    loss = []
    # print(col_i)
    for season in sub['Season'].unique():
        # print(season)
        x1 = games.loc[((games['Season'] < int(season)))]
        x2 = games.loc[((games['Season'] == int(season)))]
        test = sub[sub['Season'] == season]
        # reg = linear_model.HuberRegressor()
        # reg = linear_model.Ridge()
        reg = LogisticRegression()
        reg.fit(x1[col_i], x1['result'])
        pred = reg.predict(x2[col_i]).clip(0.05, 0.95)
        # print('Log Loss:', metrics.log_loss(x2['result'], pred))
        loss.append(metrics.log_loss(x2['result'], pred))
    loss = np.mean(loss)
    print(col[i], 'Total Log Loss:', loss)
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
Seed
2014
Log Loss: 0.631103496715
2015
Log Loss: 0.536077947043
2016
Log Loss: 0.613942108476
2017
Log Loss: 0.524112484252
Testing for Sequence of Scoring
Total Log Loss: 0.576309009122

Seed OridinalRank
2014
Log Loss: 0.640362499596
2015
Log Loss: 0.535640531797
2016
Log Loss: 0.615240856774
2017
Log Loss: 0.526454174234

OridinalRank
2014
Log Loss: 0.569626376883
2015
Log Loss: 0.588759741543
2016
Log Loss: 0.610062379138
2017
Log Loss: 0.548454259202
Testing for Sequence of Scoring
Total Log Loss: 0.579225689191

step wise
['Seed', 'OrdinalRank', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.553149692947
OrdinalRank Total Log Loss: 0.551839570571
PIE Total Log Loss: 0.554426941981
FG_PCT Total Log Loss: 0.554782674664
TURNOVER_RATE Total Log Loss: 0.553026616351
OFF_REB_PCT Total Log Loss: 0.556224397715
FT_RATE Total Log Loss: 0.559941715051
4FACTOR Total Log Loss: 0.552939118782
OFF_EFF Total Log Loss: 0.562279798327
DEF_EFF Total Log Loss: 0.557964855392
ASSIST_RATIO Total Log Loss: 0.554418056889
DEF_REB_PCT Total Log Loss: 0.550722224597******************
FT_PCT Total Log Loss: 0.553010125845
WINPCT Total Log Loss: 0.56285734474

['Seed', 'OrdinalRank', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.551755732165
OrdinalRank Total Log Loss: 0.549458202083*****************
PIE Total Log Loss: 0.554457874259
FG_PCT Total Log Loss: 0.553598643825
TURNOVER_RATE Total Log Loss: 0.558484878696
OFF_REB_PCT Total Log Loss: 0.555443833417
FT_RATE Total Log Loss: 0.558245078574
4FACTOR Total Log Loss: 0.554231883232
OFF_EFF Total Log Loss: 0.560549442413
DEF_EFF Total Log Loss: 0.55607875307
ASSIST_RATIO Total Log Loss: 0.553756909556
FT_PCT Total Log Loss: 0.551029290518
WINPCT Total Log Loss: 0.561190393871

['Seed', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.590373054626
PIE Total Log Loss: 0.545199850467
FG_PCT Total Log Loss: 0.549672393247
TURNOVER_RATE Total Log Loss: 0.550315217774
OFF_REB_PCT Total Log Loss: 0.548460991632
FT_RATE Total Log Loss: 0.551429600423
4FACTOR Total Log Loss: 0.550179312138
OFF_EFF Total Log Loss: 0.55034612156
DEF_EFF Total Log Loss: 0.54509848036*****************
ASSIST_RATIO Total Log Loss: 0.550182715665
FT_PCT Total Log Loss: 0.548876113304
WINPCT Total Log Loss: 0.556286876803
Testing for Sequence of Scoring
Total Log Loss: 0.556286876803

['Seed', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', '4FACTOR', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.584686416691
PIE Total Log Loss: 0.562437784229
FG_PCT Total Log Loss: 0.546862443906
TURNOVER_RATE Total Log Loss: 0.546444745174
OFF_REB_PCT Total Log Loss: 0.548268109482
FT_RATE Total Log Loss: 0.551860762265
4FACTOR Total Log Loss: 0.546177736771**************
OFF_EFF Total Log Loss: 0.54963598774
ASSIST_RATIO Total Log Loss: 0.547374753359
FT_PCT Total Log Loss: 0.546998081303
WINPCT Total Log Loss: 0.55209648054

['Seed', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.585108642668
PIE Total Log Loss: 0.562177091128
FG_PCT Total Log Loss: 0.54498771819******************
TURNOVER_RATE Total Log Loss: 0.546368706322
OFF_REB_PCT Total Log Loss: 0.54595189243
FT_RATE Total Log Loss: 0.558130073998
OFF_EFF Total Log Loss: 0.546836462802
ASSIST_RATIO Total Log Loss: 0.54712278067
FT_PCT Total Log Loss: 0.546276501278
WINPCT Total Log Loss: 0.549808792311

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
Seed Total Log Loss: 0.583597292164
PIE Total Log Loss: 0.562815543585
TURNOVER_RATE Total Log Loss: 0.547415516537
OFF_REB_PCT Total Log Loss: 0.551947923752
FT_RATE Total Log Loss: 0.557076246728
OFF_EFF Total Log Loss: 0.546082559359**************
ASSIST_RATIO Total Log Loss: 0.547513441248
FT_PCT Total Log Loss: 0.546474449369
WINPCT Total Log Loss: 0.549096423858



"""