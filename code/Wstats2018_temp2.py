import pandas as pd
import numpy as np
from sklearn import *
import os, glob
from sklearn.linear_model import LogisticRegression

#Generate a list of all matchups in the tourney since 2003

#df_tourney_list = pd.read_csv('NCAATourneyCompactResults.csv')
df_tourney_list = pd.read_csv('../Winput/WNCAATourneyCompactResults_PrelimData2018.csv')
df_tourney_list.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tourney_list = df_tourney_list[df_tourney_list['Season'] > 2009]
df_tourney_list.reset_index(inplace = True, drop=True)
df_tourney_list.head()

# load feature
df_tourney_final = pd.read_csv("../additional/Wdf_tourney_final_temp2018.csv")

# make train data
#gets the features for the winning team

df_model_winners = pd.merge(left=df_tourney_list, right=df_tourney_final ,how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
df_model_winners.drop(labels=['TeamID'], inplace=True, axis=1)
print(df_model_winners.head())

#gets the features for the losing team

df_model_losers = pd.merge(left=df_tourney_list, right=df_tourney_final ,how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
df_model_losers.drop(labels=['TeamID'], inplace=True, axis=1)
df_model_losers.head()

#This generates the differences between the features between winning and losing team and assigns 1 as the classifier for winning

df_model_winner_diff = (df_model_winners.iloc[:, 3:] - df_model_losers.iloc[:, 3:])
df_model_winner_diff['result'] = 1
df_model_winner_diff = pd.merge(left=df_model_winner_diff, right=df_tourney_list, left_index=True, right_index=True, how='inner')

#This generates the differences between the features between losing and winning team and assigns 0 as the classifier for losing

df_model_loser_diff = (df_model_losers.iloc[:, 3:] - df_model_winners.iloc[:, 3:])
df_model_loser_diff['result'] = 0
df_model_loser_diff = pd.merge(left=df_model_loser_diff, right=df_tourney_list, left_index=True, right_index=True, how='inner')

df_predictions_tourney = pd.concat((df_model_winner_diff, df_model_loser_diff), axis=0)

df_predictions_tourney.sort_values('Season', inplace=True)

df_predictions_tourney.reset_index(inplace = True, drop=True)
games = df_predictions_tourney
games.to_csv("../additional/Wdf_predictions_tourney_2018_temp.csv", index=None)
print(games['Season'].dtype)

# Test Set
print("Test Set")
sub = pd.read_csv('../Winput/WSampleSubmissionStage2_SampleTourney2018.csv')
sub['Season'] = sub['ID'].map(lambda x: int(x.split('_')[0]))
sub['WTeamID'] = sub['ID'].map(lambda x: int(x.split('_')[1]))
sub['LTeamID'] = sub['ID'].map(lambda x: int(x.split('_')[2]))


sub_model_winners = pd.merge(left=sub, right=df_tourney_final ,how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
sub_model_winners.drop(labels=['TeamID'], inplace=True, axis=1)
print(sub_model_winners.columns)

print(sub['Season'].dtype)
print(sub['WTeamID'].dtype)

#gets the features for the losing team

sub_model_losers = pd.merge(left=sub[['Season', 'LTeamID']], right=df_tourney_final ,how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
sub_model_losers.drop(labels=['TeamID'], inplace=True, axis=1)
sub_model_losers.tail()

print(sub_model_winners.head())
#This generates the differences between the features between winning and losing team and assigns 1 as the classifier for winning
print("sub_model_winner_diff")
sub_model_winner_diff = (sub_model_winners.iloc[:, :] - sub_model_losers.iloc[:, :])
sub_model_winner_diff = pd.merge(left=sub_model_winner_diff, right=df_tourney_list, left_index=True, right_index=True, how='left')
sub_model_winner_diff['Season'] = sub_model_winners['Season']
print(sub_model_winner_diff.head())
print(sub_model_winner_diff.columns)
#
# #This generates the differences between the features between losing and winning team and assigns 0 as the classifier for losing
#
sub_model_loser_diff = (sub_model_losers.iloc[:, 3:] - sub_model_winners.iloc[:, 3:])
sub_model_loser_diff = pd.merge(left=sub_model_loser_diff, right=df_tourney_list, left_index=True, right_index=True, how='left')
sub_model_loser_diff['Season'] = sub_model_losers['Season']
print("sub_model_winner_diff.shape", sub_model_winner_diff.shape)
print("sub.shape", sub.shape)









# Add Validation
print("Add Validation")
results = []
loss = []
col = ['Seed',
       'PIE',
       'TURNOVER_RATE', 'OFF_REB_PCT',
       # 'FT_RATE', # wrong
       'OFF_EFF',
       'ASSIST_RATIO', # wrong
       'FT_PCT', # wrong
       'WINPCT',
       'experience'# wrong
       ]
# col = ['Seed','PIE']
loss = []
# print(col_i)
train_season = [2014, 2015, 2016, 2017]
col = [
    'Seed',
    # 'PIE',
    'FG_PCT',
    'TURNOVER_RATE',
    'OFF_REB_PCT',
    # 'FT_RATE',
    '4FACTOR',
    'OFF_EFF',
    'DEF_EFF',
    # 'ASSIST_RATIO',
    'DEF_REB_PCT',
    'FT_PCT',
    'WINPCT',
    'experience',
    # 'result',
    # 'Season',
    # 'WTeamID',
    # 'LTeamID',
]
print(col)
for i in range(len(col)):
    col_i = col[:i] + col[i+1:]
    # col_i = col
    loss = []
    # print(col_i)
    for season in train_season:
        # print(season)
        x1 = games.loc[((games['Season'] < int(season)))]
        x2 = games.loc[((games['Season'] == int(season)))]
        test = sub[sub['Season'] == season]
        reg = linear_model.HuberRegressor()
        # reg = linear_model.Ridge()
        # reg = LogisticRegression()
        reg.fit(x1[col_i], x1['result'])
        pred = reg.predict(x2[col_i]).clip(0.05, 0.95)
        # print('Log Loss:', metrics.log_loss(x2['result'], pred))
        loss.append(metrics.log_loss(x2['result'], pred))
    loss = np.mean(loss)
    print(col[i], 'Total Log Loss:', loss)
"""
['Seed', 'PIE', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', 'OFF_EFF', 'ASSIST_RATIO', 'FT_PCT', 'WINPCT', 'experience']
2014
Log Loss: 0.464771982001
2015
Log Loss: 0.386960388031
2016
Log Loss: 0.479512140405
2017
Log Loss: 0.444736949188
Total Log Loss: 0.443995364906
2018
[ 0.56996335  0.07728762  0.48999043 ...,  0.40467298  0.79589589
  0.89110811]



['Seed', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', 'FT_RATE', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
Seed Total Log Loss: 0.498921406095
PIE Total Log Loss: 0.431284814573
FG_PCT Total Log Loss: 0.433581527307
TURNOVER_RATE Total Log Loss: 0.43385755652
OFF_REB_PCT Total Log Loss: 0.431224687709
FT_RATE Total Log Loss: 0.43082555785*******************
4FACTOR Total Log Loss: 0.433569239927
OFF_EFF Total Log Loss: 0.436660386744
DEF_EFF Total Log Loss: 0.441617879607
ASSIST_RATIO Total Log Loss: 0.431002721347
DEF_REB_PCT Total Log Loss: 0.43420205043
FT_PCT Total Log Loss: 0.438024380082
WINPCT Total Log Loss: 0.446789586162
experience Total Log Loss: 0.431954766731


['Seed', 'PIE', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
Seed Total Log Loss: 0.499382239853
PIE Total Log Loss: 0.428186635013******************
FG_PCT Total Log Loss: 0.429992996017
TURNOVER_RATE Total Log Loss: 0.430316819252
OFF_REB_PCT Total Log Loss: 0.428869603653
4FACTOR Total Log Loss: 0.430879230614
OFF_EFF Total Log Loss: 0.431180866083
DEF_EFF Total Log Loss: 0.440639271029
ASSIST_RATIO Total Log Loss: 0.430018369566
DEF_REB_PCT Total Log Loss: 0.432009189249
FT_PCT Total Log Loss: 0.43559097455
WINPCT Total Log Loss: 0.447331288298
experience Total Log Loss: 0.431978579616

['Seed', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'ASSIST_RATIO', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
Seed Total Log Loss: 0.503577468994
FG_PCT Total Log Loss: 0.42677814192
TURNOVER_RATE Total Log Loss: 0.428313336972
OFF_REB_PCT Total Log Loss: 0.42666423418
4FACTOR Total Log Loss: 0.42734655017
OFF_EFF Total Log Loss: 0.429899672531
DEF_EFF Total Log Loss: 0.45184148941
ASSIST_RATIO Total Log Loss: 0.42643101009*************** best
DEF_REB_PCT Total Log Loss: 0.431336600105
FT_PCT Total Log Loss: 0.435567482643
WINPCT Total Log Loss: 0.444638981077
experience Total Log Loss: 0.430851100437

['Seed', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
Seed Total Log Loss: 0.5023607135
FG_PCT Total Log Loss: 0.426979407925
TURNOVER_RATE Total Log Loss: 0.426816364139
OFF_REB_PCT Total Log Loss: 0.42652527007
4FACTOR Total Log Loss: 0.427151759913
OFF_EFF Total Log Loss: 0.428044110146
DEF_EFF Total Log Loss: 0.452399416915
DEF_REB_PCT Total Log Loss: 0.429176197496
FT_PCT Total Log Loss: 0.432400413563
WINPCT Total Log Loss: 0.44404261815
experience Total Log Loss: 0.428503169112
"""