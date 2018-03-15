# MarchMadness

最終サブミッション処理手順  
Men  
`code/experience.ipynb` : 特徴量 experience を作成 (過去3年のNCAA経験試合数)  
`code/Mstats_2018.ipynb` : その他の特徴量生成 (スタッツ, シード, ランキング等)  
`code/Mstats_2018.py` : 予測 (Huber 70% Ridge 30%)  

Women  
`code/Wexperience.ipynb` : 特徴量 experience を作成 (過去3年のNCAA経験試合数)  
`code/Wstats_2018.ipynb` : その他の特徴量生成 (スタッツ, シード, ランキング等)  
`code/Wstats_2018.py` : 予測 (Huber 70% Ridge 30%)  

データ置き場  
`Minput2/` : `Stage2UnchangedDataFiles.zip`内のファイルと`WSampleSubmissionStage2.csv`を置く  
`Minput3/` : `Stage2UpdatedDataFiles.zip`内のファイルを置く  
`Winput2/` : `WStage2DataFiles.zip`内のファイルと`WSampleSubmissionStage2.csv`を置く  