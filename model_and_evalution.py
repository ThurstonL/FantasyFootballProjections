import pandas as pd
from dataPrep import qb_split, rb_split, wr_split, te_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

#Will use 2 common regression algorithms (Linear Regression and Random Forest Regression)

#Running test_train_splitting functions 
qb_x_train, qb_x_test, qb_y_train, qb_y_test = qb_split()
rb_x_train, rb_x_test, rb_y_train, rb_y_test = rb_split()
wr_x_train, wr_x_test, wr_y_train, wr_y_test = wr_split()
te_x_train, te_x_test, te_y_train, te_y_test = te_split()

#Linear Regression
qb_reg_model = LinearRegression()
qb_reg_model.fit(qb_x_train, qb_y_train)
qb_y_reg_pred_test = qb_reg_model.predict(qb_x_test)
qb_reg_pred_test_df = pd.DataFrame({'Actual': qb_y_test, 'Predicted': qb_y_reg_pred_test})
qb_r2_reg_model_test = round(qb_reg_model.score(qb_x_test, qb_y_test), 2) #Calculating R^2 value
print(f'QB Linear Regression Model R^2 value: {qb_r2_reg_model_test}')

rb_reg_model = LinearRegression()
rb_reg_model.fit(rb_x_train, rb_y_train)
rb_y_reg_pred_test = rb_reg_model.predict(rb_x_test)
rb_reg_pred_test_df = pd.DataFrame({'Actual': rb_y_test, 'Predicted': rb_y_reg_pred_test})
rb_r2_reg_model_test = round(rb_reg_model.score(rb_x_test, rb_y_test), 2) #Calculating R^2 value
print(f'RB Linear Regression Model R^2 value: {rb_r2_reg_model_test}')

wr_reg_model = LinearRegression()
wr_reg_model.fit(wr_x_train, wr_y_train)
wr_y_reg_pred_test = wr_reg_model.predict(wr_x_test)
wr_reg_pred_test_df = pd.DataFrame({'Actual': wr_y_test, 'Predicted': wr_y_reg_pred_test})
wr_r2_reg_model_test = round(wr_reg_model.score(wr_x_test, wr_y_test), 2) #Calculating R^2 value
print(f'WR Linear Regression Model R^2 value: {wr_r2_reg_model_test}')

te_reg_model = LinearRegression()
te_reg_model.fit(te_x_train, te_y_train)
te_y_reg_pred_test = te_reg_model.predict(te_x_test)
te_reg_pred_test_df = pd.DataFrame({'Actual': te_y_test, 'Predicted': te_y_reg_pred_test})
te_r2_reg_model_test = round(te_reg_model.score(te_x_test, te_y_test), 2) #Calculating R^2 value
print(f'TE Linear Regression Model R^2 value: {te_r2_reg_model_test}')

#Random Forest Regressor
qb_rf_model = RandomForestRegressor(n_estimators=10, random_state=10)
qb_rf_model.fit(qb_x_train, qb_y_train)
qb_y_rf_pred_test = qb_rf_model.predict(qb_x_test)
qb_rf_pred_test_df = pd.DataFrame({'Actual': qb_y_test, 'Predicted': qb_y_rf_pred_test})
qb_r2_rf = r2_score(qb_y_test, qb_y_rf_pred_test)
print(f'QB Random Forest Model R^2: {round(qb_r2_rf,2)*100}')
print(f"RMSE for QB Random Forest Model: {(mean_squared_error(qb_rf_pred_test_df['Actual'],qb_rf_pred_test_df['Predicted']))**0.5}")

rb_rf_model = RandomForestRegressor(n_estimators=10, random_state=10)
rb_rf_model.fit(rb_x_train, rb_y_train)
rb_y_rf_pred_test = rb_rf_model.predict(rb_x_test)
rb_rf_pred_test_df = pd.DataFrame({'Actual': rb_y_test, 'Predicted': rb_y_rf_pred_test})
rb_r2_rf = r2_score(rb_y_test, rb_y_rf_pred_test)
print(f'RB Random Forest Model R^2: {round(rb_r2_rf,2)*100}')
print(f"RMSE for RB Random Forest Model: {(mean_squared_error(rb_rf_pred_test_df['Actual'],rb_rf_pred_test_df['Predicted']))**0.5}")

wr_rf_model = RandomForestRegressor(n_estimators=10, random_state=10)
wr_rf_model.fit(wr_x_train, wr_y_train)
wr_y_rf_pred_test = wr_rf_model.predict(wr_x_test)
wr_rf_pred_test_df = pd.DataFrame({'Actual': wr_y_test, 'Predicted': wr_y_rf_pred_test})
wr_r2_rf = r2_score(wr_y_test, wr_y_rf_pred_test)
print(f'WR Random Forest Model R^2: {round(wr_r2_rf,2)*100}')
print(f"RMSE for WR Random Forest Model: {(mean_squared_error(wr_rf_pred_test_df['Actual'],wr_rf_pred_test_df['Predicted']))**0.5}")

te_rf_model = RandomForestRegressor(n_estimators=10, random_state=10)
te_rf_model.fit(te_x_train, te_y_train)
te_y_rf_pred_test = te_rf_model.predict(te_x_test)
te_rf_pred_test_df = pd.DataFrame({'Actual': te_y_test, 'Predicted': te_y_rf_pred_test})
te_r2_rf = r2_score(te_y_test, te_y_rf_pred_test)
print(f'TE Random Forest Model R^2: {round(te_r2_rf,2)*100}')
print(f"RMSE for TE Random Forest Model: {(mean_squared_error(te_rf_pred_test_df['Actual'],te_rf_pred_test_df['Predicted']))**0.5}")

#Final Projections using Linear Regression since it modeled the data better
qb_stats = pd.read_csv('QB2013to2023.csv')
qb_stats_2023 = qb_stats[qb_stats['Year']==2023]
qb_stats_2023_test = qb_stats_2023[['Year', 'InjuryFactor', 'PassEfficiency', 'PassTD', 'RushEfficiency', 'RushTD', 'PPR']]
qb_stats_2024_pred = qb_reg_model.predict(qb_stats_2023_test)
qb_stats_2023.loc[:,'NextPPR'] = qb_stats_2024_pred
qb_stats_2023.to_csv('QB_Linear_Regression_Predictions.csv')

rb_stats = pd.read_csv('RB2013to2023.csv')
rb_stats_2023 = rb_stats[rb_stats['Year']==2023]
rb_stats_2023_test = rb_stats_2023[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
rb_stats_2024_pred = rb_reg_model.predict(rb_stats_2023_test)
rb_stats_2023.loc[:,'NextPPR'] = rb_stats_2024_pred
rb_stats_2023.to_csv('RB_Linear_Regression_Predictions.csv')

wr_stats = pd.read_csv('WR2013to2023.csv')
wr_stats_2023 = wr_stats[wr_stats['Year']==2023]
wr_stats_2023_test = wr_stats_2023[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
wr_stats_2024_pred = wr_reg_model.predict(wr_stats_2023_test)
wr_stats_2023.loc[:,'NextPPR'] = wr_stats_2024_pred
wr_stats_2023.to_csv('WR_Linear_Regression_Predictions.csv')

te_stats = pd.read_csv('TE2013to2023.csv')
te_stats_2023 = te_stats[te_stats['Year']==2023]
te_stats_2023_test = te_stats_2023[['Year', 'InjuryFactor', 'RushEfficiency', 'RecEfficiency', 'TotTD', 'PPR']]
te_stats_2024_pred = te_reg_model.predict(te_stats_2023_test)
te_stats_2023.loc[:,'NextPPR'] = te_stats_2024_pred
te_stats_2023.to_csv('RB_Linear_Regression_Predictions.csv')