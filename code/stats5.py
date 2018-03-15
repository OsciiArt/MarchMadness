import pandas as pd
import numpy as np
from sklearn import *
import os, glob
from sklearn.linear_model import LogisticRegression

datafiles = sorted(glob.glob('../input/**.csv'))
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
       'DEF_REB_PCT', 'FT_PCT', 'WINPCT',
       # 'experience',
       'result', 'Season',
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
col = ['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT',
       'FT_RATE',
       'OFF_EFF',
       'ASSIST_RATIO',
       'FT_PCT',
       'WINPCT',
       # 'experience'
       ]
# col = ['Seed','PIE']
# col2 = ['Seed',
#         # 'PIE',
#         ]
print(col)
# print(col2)
loss = []
# print(col_i)
for season in sub['Season'].unique():
    print(season)
    x1 = games.loc[((games['Season'] < int(season)))]
    x2 = games.loc[((games['Season'] == int(season)))]
    test = sub[sub['Season'] == season]
    reg = linear_model.HuberRegressor()
    reg2 = linear_model.Ridge()
    reg3 = linear_model.LogisticRegression()
    reg.fit(x1[col], x1['result'])
    reg2.fit(x1[col], x1['result'])
    # reg3.fit(x1[col2], x1['result'])
    pred = reg.predict(x2[col])
    pred2 = reg2.predict(x2[col])
    # pred3 = reg3.predict(x2[col2])
    pred = (0.7 * pred + 0.3 * pred2 ).clip(0.05, 0.95)
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
huber0.7 + ridge 0.3
2014
Log Loss: 0.583560032646
2015
Log Loss: 0.50390661504
2016
Log Loss: 0.570404157269
2017
Log Loss: 0.52054376832
Total Log Loss: 0.544603643319
Testing for Sequence of Scoring
Total Log Loss: 0.544603643319

clip(0.1, 0.9)
2014
Log Loss: 0.591629767164
2015
Log Loss: 0.511305808207
2016
Log Loss: 0.566725492793
2017
Log Loss: 0.527806529386
Total Log Loss: 0.549366899388

clip(0.01, 0.99)
2014
Log Loss: 0.578229392361
2015
Log Loss: 0.499337399535
2016
Log Loss: 0.59132821422
2017
Log Loss: 0.515545300267
Total Log Loss: 0.546110076596


add experience
Log Loss: 0.5819553038268845
2015
Log Loss: 0.5127859454486615
2016
Log Loss: 0.5685624700520782
2017
Log Loss: 0.511362236118586
Total Log Loss: 0.5436664888615526
Testing for Sequence of Scoring
Total Log Loss: 0.5436664888615526


add experience.clip(0.05, 0.95)
(0.7 * pred + 0.3 * pred2
2014
Log Loss: 0.579876717545
2015
Log Loss: 0.511582990342
2016
Log Loss: 0.568852205728
2017
Log Loss: 0.515923749039
Total Log Loss: 0.544058915664
Testing for Sequence of Scoring
Total Log Loss: 0.544058915664

col = ['Seed','PIE']
2014
Log Loss: 0.610096079946
2015
Log Loss: 0.52083541885
2016
Log Loss: 0.588178328362
2017
Log Loss: 0.521925061498
Total Log Loss: 0.560258722164
Testing for Sequence of Scoring
Total Log Loss: 0.560258722164


['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT']
2014
Log Loss: 0.602759754724
2015
Log Loss: 0.509618123385
2016
Log Loss: 0.598903245609
2017
Log Loss: 0.5092372017
Total Log Loss: 0.555129581354
Testing for Sequence of Scoring
Total Log Loss: 0.555129581354


['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
2014
Log Loss: 0.580810057775
2015
Log Loss: 0.503942577314
2016
Log Loss: 0.570310973935
2017
Traceback (most recent call last):
Log Loss: 0.520552645198
  File "C:/Users/osaka2/Documents/MEGA/Python/MarchMadness/code/stats5.py", line 138, in <module>
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
TypeError: list indices must be integers or slices, not list
Total Log Loss: 0.543904063555
Testing for Sequence of Scoring
Total Log Loss: 0.543904063555

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
2014
Log Loss: 0.598242072556
2015
Log Loss: 0.512755367536
2016
Traceback (most recent call last):
  File "C:/Users/osaka2/Documents/MEGA/Python/MarchMadness/code/stats5.py", line 139, in <module>
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
TypeError: list indices must be integers or slices, not list
Log Loss: 0.598516236923
2017
Log Loss: 0.511674739729
Total Log Loss: 0.555297104186
Testing for Sequence of Scoring
Total Log Loss: 0.555297104186

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'ASSIST_RATIO']
2014
Log Loss: 0.601365069698
2015
Log Loss: 0.509986502008
2016
Log Loss: 0.603609885333
2017
Traceback (most recent call last):
  File "C:/Users/osaka2/Documents/MEGA/Python/MarchMadness/code/stats5.py", line 140, in <module>
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
TypeError: list indices must be integers or slices, not list
Log Loss: 0.510433856402
Total Log Loss: 0.55634882836
Testing for Sequence of Scoring
Total Log Loss: 0.55634882836

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_PCT']
2014
Log Loss: 0.600206239556
2015
Log Loss: 0.5139578713
2016
Log Loss: 0.594425082628
2017
Log Loss: 0.513935413946
Total Log Loss: 0.555631151858
Testing for Sequence of Scoring
Total Log Loss: 0.555631151858


['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_PCT', 'experience']
2014
Log Loss: 0.598620842501
2015
Log Loss: 0.517526315921
2016
Log Loss: 0.591170236914
2017
Traceback (most recent call last):
  File "C:/Users/osaka2/Documents/MEGA/Python/MarchMadness/code/stats5.py", line 141, in <module>
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
TypeError: list indices must be integers or slices, not list
Log Loss: 0.504262272915
Total Log Loss: 0.552894917063
Testing for Sequence of Scoring
Total Log Loss: 0.552894917063

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'experience']
2014
Log Loss: 0.600815640458
2015
Log Loss: 0.513583889186
2016
Log Loss: 0.596573260726
2017
Traceback (most recent call last):
Log Loss: 0.500750800348
  File "C:/Users/osaka2/Documents/MEGA/Python/MarchMadness/code/stats5.py", line 141, in <module>
Total Log Loss: 0.55293089768
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
Testing for Sequence of Scoring
Total Log Loss: 0.55293089768


['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE']
2014
Log Loss: 0.595423917159
2015
Log Loss: 0.504858903942
2016
Log Loss: 0.577633701581
2017
Log Loss: 0.51073695994
Total Log Loss: 0.547163370655
Testing for Sequence of Scoring
Total Log Loss: 0.547163370655

['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT']
    results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
2014
Log Loss: 0.580810057775
2015
Log Loss: 0.503942577314
2016
Log Loss: 0.570310973935
2017
Log Loss: 0.520552645198
Total Log Loss: 0.543904063555
Testing for Sequence of Scoring
Total Log Loss: 0.543904063555

"""