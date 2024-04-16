Bulldozer Price Prediction using Machine Learning
In this notebook, we'll be tackling the Kaggle competition for predicting the auction sale price of bulldozers: https://www.kaggle.com/c/bluebook-for-bulldozers/data

Table of Contents:
* Problem Definition
* Setup Environment
* Load Data
* Data Preprocessing and EDA
  * Feature Engineering
  * Convert dtype of Data
  * Fill Missing Values
* Model Experimentation with RandomForestRegressor
  * Hyperparameter Tuning using RandomizedSearchCV
* Model Experimentation with XGBRegressor for Comparison
  * Hyperparameter Tuning using RandomizedSearchCV
* Make Predicitions on Test Dataset
* Feature Importance

Problem Definition:
Given the usage and equipment configurations of a bulldozer, how well can we predict its future sale price?

Data:
The 3 datasets provided by Kaggle are described as:

* Train.csv is the training set, which contains data through the end of 2011.
* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
* Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition

Evaluation Metric:
The competition evaluates on the root mean squared log error (RMSLE) between the actual and predicted price of a sold bulldozer. We will also be taking a look at the mean absolute error (MAE) and the r^2 score.
