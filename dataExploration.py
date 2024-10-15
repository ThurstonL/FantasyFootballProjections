import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Data Exploration file. Heatmap was created to find highly correlated features. Feature engineering to reduce dimensionality and combine similar variables.


'''
    column_order = ['FantPos', 'Player', 'Year', 'Age', 'G', 'TG', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int',
                     'RushAtt', 'RushYds', 'RushTD', 'Tgt', 'Rec',
                     'RecYds', 'RecTD', 'TotTD', 'PPR', 'NextPPR']
'''

stats = pd.read_csv('2013to2023stats.csv')
#print(stats.head())

#################### GROUP BY QB #################################################

QB_stats = stats[stats['FantPos'] == 'QB']
QB_columns_to_remove = ['FantPos', 'Tgt', 'Rec', 'RecYds', 'RecTD', 'TotTD']
QB_stats = QB_stats.drop(columns=QB_columns_to_remove)

#Exploring distribution of data. Must remove player name above for heat map to work
'''
QB_stats.hist(bins=50, figsize=(16,9))
QB_corr = QB_stats.corr()
plt.figure(figsize=(8,8))
sns.heatmap(QB_corr, annot=True)
plt.show()
'''

#Feature Engineering to remove highly correlated stats
QB_stats['InjuryFactor'] = QB_stats['Age'] * QB_stats['G'] / QB_stats['TG']
QB_stats = QB_stats.drop(columns=['Age', 'G', 'TG'])
QB_stats['PassEfficiency'] = QB_stats['Cmp'] * QB_stats['PassYds'] / QB_stats['PassAtt']
QB_stats['PassEfficiency'] = QB_stats['PassEfficiency'].fillna(0)
QB_stats = QB_stats.drop(columns=['Cmp', 'PassYds', 'PassAtt'])
QB_stats['RushEfficiency'] = QB_stats['RushYds'] / QB_stats['RushAtt']
QB_stats['RushEfficiency'] = QB_stats['RushEfficiency'].fillna(0)
QB_stats.replace([np.inf, -np.inf], 0, inplace=True)
QB_stats = QB_stats.drop(columns=['RushYds', 'RushAtt'])

'''
QB_corr = QB_stats.corr()
plt.figure(figsize=(10,10))
sns.heatmap(QB_corr, annot=True)
plt.show()
'''
QB_column_order = ['Player', 'Year', 'InjuryFactor', 'PassEfficiency', 'PassTD', 'RushEfficiency', 'RushTD','PPR', 'NextPPR']
QB_stats = QB_stats.reindex(columns= QB_column_order)
QB_stats.to_csv('QB2013to2023.csv')

#################### GROUP BY RB #################################################
RB_stats = stats[stats['FantPos'] == 'RB']
RB_columns_to_remove = ['FantPos', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushTD', 'RecTD']
RB_stats = RB_stats.drop(columns=RB_columns_to_remove)

#Exploring distribution of data. Must remove player name above for heat map to work
'''
RB_stats.hist(bins=50, figsize=(16,9))
RB_corr = RB_stats.corr()
plt.figure(figsize=(8,8))
sns.heatmap(RB_corr, annot=True)
plt.show()
'''

#Feature Engineering to remove highly correlated stats
RB_stats['InjuryFactor'] = RB_stats['Age'] * RB_stats['G'] / RB_stats['TG']
RB_stats = RB_stats.drop(columns=['Age', 'G', 'TG'])
RB_stats['RecEfficiency'] = RB_stats['RecYds'] * RB_stats['Rec'] / RB_stats['Tgt']
RB_stats['RecEfficiency'] = RB_stats['RecEfficiency'].fillna(0)
RB_stats = RB_stats.drop(columns=['RecYds', 'Rec', 'Tgt'])
RB_stats['RushEfficiency'] = RB_stats['RushYds'] / RB_stats['RushAtt']
RB_stats['RushEfficiency'] = RB_stats['RushEfficiency'].fillna(0)
RB_stats.replace([np.inf, -np.inf], 0, inplace=True)
RB_stats = RB_stats.drop(columns=['RushYds', 'RushAtt'])

'''
RB_corr = RB_stats.corr()
plt.figure(figsize=(10,10))
sns.heatmap(RB_corr, annot=True)
plt.show()
'''

RB_column_order = ['Player', 'Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR', 'NextPPR']
RB_stats = RB_stats.reindex(columns= RB_column_order)
RB_stats.to_csv('RB2013to2023.csv')

#################### GROUP BY WR #################################################
WR_stats = stats[stats['FantPos'] == 'WR']
WR_columns_to_remove = ['FantPos', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushTD', 'RecTD']
WR_stats = WR_stats.drop(columns=WR_columns_to_remove)

#Feature Engineering to remove highly correlated stats
WR_stats['InjuryFactor'] = WR_stats['Age'] * WR_stats['G'] / WR_stats['TG']
WR_stats = WR_stats.drop(columns=['Age', 'G', 'TG'])
WR_stats['RecEfficiency'] = WR_stats['RecYds'] * WR_stats['Rec'] / WR_stats['Tgt']
WR_stats['RecEfficiency'] = WR_stats['RecEfficiency'].fillna(0)
WR_stats = WR_stats.drop(columns=['RecYds', 'Rec', 'Tgt'])
WR_stats['RushEfficiency'] = WR_stats['RushYds'] / WR_stats['RushAtt']
WR_stats['RushEfficiency'] = WR_stats['RushEfficiency'].fillna(0)
WR_stats.replace([np.inf, -np.inf], 0, inplace=True)
WR_stats = WR_stats.drop(columns=['RushYds', 'RushAtt'])

WR_column_order = ['Player', 'Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR', 'NextPPR']
WR_stats = WR_stats.reindex(columns= WR_column_order)
WR_stats.to_csv('WR2013to2023.csv')

#################### GROUP BY TE #################################################
TE_stats = stats[stats['FantPos'] == 'TE']
TE_columns_to_remove = ['FantPos', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'RushTD', 'RecTD']
TE_stats = TE_stats.drop(columns=TE_columns_to_remove)

#Feature Engineering to remove highly correlated stats
TE_stats['InjuryFactor'] = TE_stats['Age'] * TE_stats['G'] / TE_stats['TG']
TE_stats = TE_stats.drop(columns=['Age', 'G', 'TG'])
TE_stats['RecEfficiency'] = TE_stats['RecYds'] * TE_stats['Rec'] / TE_stats['Tgt']
TE_stats['RecEfficiency'] = TE_stats['RecEfficiency'].fillna(0)
TE_stats = TE_stats.drop(columns=['RecYds', 'Rec', 'Tgt'])
TE_stats['RushEfficiency'] = TE_stats['RushYds'] / TE_stats['RushAtt']
TE_stats['RushEfficiency'] = TE_stats['RushEfficiency'].fillna(0)
TE_stats.replace([np.inf, -np.inf], 0, inplace=True)
TE_stats = TE_stats.drop(columns=['RushYds', 'RushAtt'])

TE_column_order = ['Player', 'Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR', 'NextPPR']
TE_stats = TE_stats.reindex(columns= TE_column_order)
TE_stats.to_csv('TE2013to2023.csv')
