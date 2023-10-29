# Team LuAnMa CS-433 ML Project 1

This project code runs the best performing model: optimised logistic regression on the 2015 BRFSS dataset, outputting predictions into a submission file on [AICrowd Class Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/). All other ML implementations are in implementations.py.

The run.py outputs a model achieving an F1 score of 0.405 and an accuracy of 0.860.

## Dataset
The [2015 BRFSS dataset](https://www.cdc.gov/brfss/annual_data/annual_2015.html) contains, for each respondent, 321 survey question answers which are our data features inuluding both ordinal and nominal categorical data with varying ranges. Labels take on values of {-1, 1}, where respondents labelled with 1 have been diagnosed with coronary heart disease or have experienced myocardial infarction. 

## Data processing
To prepare the data appropriately for the ML methods, a data processing pipeline was devised consisting of: a first feature filter, data balancing, handling of missing data, a second feature filter, feature engineering and label processing. 

## Models
The ML techniques implemented include least squares regression, linear regression with gradient descent (GD), linear regression with stochastic gradient descent (SGD), linear ridge regression, logistic regression with GD, logistic regression with GD and L2-regularised logistic regression with GD. Hyperparameters were optimised using cross-validation. For each optimised model, the F1 score and accuracy was determined such that the optimised model could be identified. 

## Running the code
```bash
python3 -m run