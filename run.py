import numpy as np
import helpers

# Manually selected list of features to delete:
filter1 = ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'ASTHMA3', 'CHCSCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'DIABAGE2', 'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'NUMHHOL2', 'CPDEMO1', 'VETERAN3', 'EMPLOY1', 'CHILDREN', 'INCOME2', 'WEIGHT2', 'HEIGHT3', 'PREGNANT', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', 'USENOW3', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'STRENGTH', 'LMTJOIN3', 'ARTHDIS2', 'JOINPAIN', 'FLUSHOT6', 'FLSHTMY2', 'PNEUVAC3', 'HIVTST6', 'HIVTSTD3', 'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2', 'DOCTDIAB', 'CHKHEMO3', 'DIABEYE', 'DIABEDU', 'CAREGIV1', 'CRGVREL1', 'CRGVPRB1', 'VINOCRE2', 'CIMEMLOS', 'WTCHSALT', 'LONGWTCH', 'DRADVISE', 'ASTHMAGE', 'ASATTACK', 'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP', 'ASTHMED3', 'ASINHALR', 'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'ASPUNSAF', 'RLIVPAIN', 'RDUCHART', 'RDUCSTRK', 'ARTTODAY', 'ARTHWGT', 'ARTHEXER', 'ARTHEDU', 'TETANUS', 'HPVADVC2', 'SHINGLE2', 'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PROFEXAM', 'LENGEXAM', 'BLDSTOOL', 'HADSIGM3', 'LASTSIG3', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'SCNTMNY1', 'SCNTMEL1', 'SCNTWRK1', 'SCNTLWK1', 'SXORIENT', 'TRNSGNDR', 'EMTSUPRT', 'LSATISFY', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE', 'MISTMNT', 'ADANXEV', 'QSTLANG', 'MSCODE', '_CHISPNC', '_DUALUSE', '_RFHLTH', '_RFHYPE5', '_PRACE1', '_MRACE1', '_RACE', '_AGEG5YR', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWEK', '_RFDRHV5', '_FRTLT1', '_VEGLT1', '_TOTINDA', 'ACTIN11_', 'ACTIN21_', '_MINAC21', '_PACAT1', '_PAINDX1', '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_FLSHOT6', '_PNEUMO2', '_AIDTST3']

## 1. Load the training data into feature matrix and class labels
x_train, x_train_head, x_test, y_train, train_ids, test_ids = helpers.load_csv_data("data")

## 2. Filter the features:
# First filter
x_train_f1, x_test_f1 = helpers.first_filter(x_train, x_train_head, x_test, filter1)

# Replace NaN values with the mean of the column:
x_train_f1, x_test_f1 = helpers.replace_nan_with_median(x_train_f1, x_test_f1)

# Second filter
x_train_f2, x_test_f2 = helpers.second_filter(x_train_f1, x_test_f1)

## 3. Train the model:
# Weights from least squares
# weights, _ = implementations.least_squares(y_train, x_train_f2)

k_folds = 2
lambdas = np.logspace(-4, -1, 4)
gammas = np.logspace(-3, 0, 4)
max_iters=1000

weights, rmse, best_lambda = helpers.train_ridge_regression(y_train, x_train_f2, k_folds, lambdas, 1)
weights, loss = helpers.train_reg_logistic_regression(y_train, x_train_f2, weights, best_lambda, gammas, max_iters)
print("Results from regression:")
print("best w: " + str(weights) + " with loss " + str(loss))

# 4. Make predictions:
y_pred_norm = helpers.make_predictions(weights, x_test_f2)

# Store the predictions in a submission_file.csv in CSV format without index_label
helpers.create_csv_submission(test_ids, y_pred_norm, 'submission_file.csv')
