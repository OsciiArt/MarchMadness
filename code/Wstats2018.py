import pandas as pd
import numpy as np
from sklearn import *
import os, glob
from sklearn.linear_model import LogisticRegression

#Generate a list of all matchups in the tourney since 2003

#df_tourney_list = pd.read_csv('NCAATourneyCompactResults.csv')
df_tourney_list = pd.read_csv('../Winput2/WNCAATourneyCompactResults.csv')
df_tourney_list.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tourney_list = df_tourney_list[df_tourney_list['Season'] > 2009]
df_tourney_list.reset_index(inplace = True, drop=True)
df_tourney_list.head()

# load feature
df_tourney_final = pd.read_csv("../additional/Wdf_tourney_final_2018.csv")

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
games.to_csv("../additional/Wdf_predictions_tourney_2018.csv", index=None)
print(games['Season'].dtype)

# Test Set
print("Test Set")
sub = pd.read_csv('../Winput2/WSampleSubmissionStage2.csv')
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
col = ['Seed', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
# col = ['Seed','PIE']
loss = []
# print(col_i)
train_season = [2014, 2015, 2016, 2017]
print(col)
for season in train_season:
    print(season)
    x1 = games.loc[((games['Season'] < int(season)))]
    x2 = games.loc[((games['Season'] == int(season)))]

    reg = linear_model.HuberRegressor()
    reg2 = linear_model.Ridge()
    reg.fit(x1[col], x1['result'])
    reg2.fit(x1[col], x1['result'])
    pred = reg.predict(x2[col])
    pred2 = reg2.predict(x2[col])
    pred = (0.7 * pred + 0.3 * pred2 ).clip(0.05, 0.95)
    print('Log Loss:', metrics.log_loss(x2['result'], pred))
    loss.append(metrics.log_loss(x2['result'], pred))
loss = np.mean(loss)
print('Total Log Loss:', loss)
season = 2018
print(season)
x1 = games.loc[((games['Season'] < int(season)))]
test = sub_model_winner_diff
reg = linear_model.HuberRegressor()
reg2 = linear_model.Ridge()
reg.fit(x1[col], x1['result'])
reg2.fit(x1[col], x1['result'])
pred = reg.predict(test[col])
pred2 = reg2.predict(test[col])
pred = (0.7 * pred + 0.3 * pred2 ).clip(0.05, 0.95)
print(pred)
print("pred.shape", pred.shape)
print("sub.shape", sub.shape)
# test['Pred'] = pred
# results = test
# print(test['Pred'])
# print(results.columns)
# results = {k: float(v) for k, v in results[['ID', 'Pred']].values}
# print(results)
sub['Pred'] = pd.DataFrame(pred)
# sub['Pred'] = sub['ID'].map(results).clip(0.05, 0.95).fillna(0.49)
# sub[['ID', 'Pred']].to_csv('../output/Wstats2018_1.csv', index=False)

sub[['ID', 'Pred']].to_csv('../output/Wstats_2018_AB.csv', index=False)
block = pd.read_csv("../Winput2/WNCAATourneySeeds.csv")
block = block.loc[(block.Season==2018)]
def blockAB(TeamID, block):
    if TeamID in block['TeamID'].as_matrix().tolist():
        Seeds = block.loc[(block.TeamID==TeamID)]['Seed'].iloc[0]
        if "Y" in Seeds or 'Z' in Seeds:
            return 'A'
        elif "W" in Seeds or 'X' in Seeds:
            return 'B'
    else:
        return 0

sub['WBlock'] = sub['WTeamID'].apply(lambda x: blockAB(x, block))
sub['LBlock'] = sub['LTeamID'].apply(lambda x: blockAB(x, block))
subA = sub.copy()
subA['Pred'] = subA.apply(lambda x: 0 if (x['WBlock']=='A' and x['LBlock']=='B') else x['Pred'], axis=1)
subA['Pred'] = subA.apply(lambda x: 1 if (x['WBlock']=='B' and x['LBlock']=='A') else x['Pred'], axis=1)
subA[['ID', 'Pred']].to_csv('../output/Wstats_2018_A.csv', index=False)

subB = sub.copy()
subB['Pred'] = subB.apply(lambda x: 0 if (x['WBlock']=='B' and x['LBlock']=='A') else x['Pred'], axis=1)
subB['Pred'] = subB.apply(lambda x: 1 if (x['WBlock']=='A' and x['LBlock']=='B') else x['Pred'], axis=1)
subB[['ID', 'Pred']].to_csv('../output/Wstats_2018_B.csv', index=False)















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


['Seed', 'FG_PCT', 'TURNOVER_RATE', 'OFF_REB_PCT', '4FACTOR', 'OFF_EFF', 'DEF_EFF', 'DEF_REB_PCT', 'FT_PCT', 'WINPCT', 'experience']
2014
Log Loss: 0.444259658034
2015
Log Loss: 0.377213268112
2016
Log Loss: 0.452036328985
2017
Log Loss: 0.434967209796
Total Log Loss: 0.427119116232
2018
[ 0.60311853  0.17987742  0.51286952 ...,  0.4647525   0.85903708
  0.89424515]
pred.shape (2016,)
sub.shape (2016, 5)
"""