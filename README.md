## Objective
Personal project with the objective of predicting the PPR fantasy points for the upcoming season using stats from the previous season.
Uses linear regression and random forest regressor models to predict stats.

## About the Files
The first file to run would be *dataCollection.py*. This file will scrape the pro football reference website for stats using
beautiful soup and convert it to a csv file. Since the files are already uploaded, there is no need to rerun. The second 
file to run would be *dataExploration.py*. This file divides all of the players by positions and creates separate csv files since 
they wouldn't necessarily use the same learning model. Heat maps are commented out, but they provide insights into stats that have 
high and low correlations with the target stat NextPPR. Highly correlated variables with each other were combined to reduce dimensionality.
The third file to run would be the *model_and_evaluation.py* file. The file calls the *dataPrep.py* functions to split
the data in test and train data. That data is then fed into their own position model for Linear Regression and Random
Forest Regressor. The models are then evaluated for R^2 score and RMSE. Finally, the predicted stats for the 2024 season
are created in a new csv file.

