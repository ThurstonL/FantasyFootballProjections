import pandas as pd
from sklearn.model_selection import train_test_split

#############Splitting QB data into test and training
def qb_split():
    qb_stats = pd.read_csv('QB2013to2023.csv')
    qb_stats = qb_stats[qb_stats['Year']!=2023]
    qb_x = qb_stats[['Year', 'InjuryFactor', 'PassEfficiency', 'PassTD', 'RushEfficiency', 'RushTD', 'PPR']]
    qb_y = qb_stats['NextPPR']
    qb_x_train, qb_x_test, qb_y_train, qb_y_test = train_test_split(qb_x, qb_y, test_size=0.2, random_state=42)
    
    return qb_x_train, qb_x_test, qb_y_train, qb_y_test

#############Splitting rb data into test and training
def rb_split():
    rb_stats = pd.read_csv('RB2013to2023.csv')
    rb_stats = rb_stats[rb_stats['Year']!=2023]
    rb_x = rb_stats[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
    rb_y = rb_stats['NextPPR']
    rb_x_train, rb_x_test, rb_y_train, rb_y_test = train_test_split(rb_x, rb_y, test_size=0.2, random_state=42)

    return rb_x_train, rb_x_test, rb_y_train, rb_y_test

#############Splitting wr data into test and training
def wr_split():
    wr_stats = pd.read_csv('WR2013to2023.csv')
    wr_stats = wr_stats[wr_stats['Year']!=2023]
    wr_x = wr_stats[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
    wr_y = wr_stats['NextPPR']
    wr_x_train, wr_x_test, wr_y_train, wr_y_test = train_test_split(wr_x, wr_y, test_size=0.2, random_state=42)

    return wr_x_train, wr_x_test, wr_y_train, wr_y_test

#############Splitting te data into test and training
def te_split():
    te_stats = pd.read_csv('TE2013to2023.csv')
    te_stats = te_stats[te_stats['Year']!=2023]
    te_x = te_stats[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
    te_y = te_stats['NextPPR']
    te_x_train, te_x_test, te_y_train, te_y_test = train_test_split(te_x, te_y, test_size=0.2, random_state=42)

    return te_x_train, te_x_test, te_y_train, te_y_test