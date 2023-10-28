import numpy as np
import helpers
import implementations

np.random.seed(42)

# Manually selected list of features to delete:
filter1 = [
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BPMEDS",
    "BPMEDS",
    "BLOODCHO",
    "CHOLCHK",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "ASTHMA3",
    "CHCSCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "DIABAGE2",
    "SEX",
    "MARITAL",
    "EDUCA",
    "RENTHOM1",
    "NUMHHOL2",
    "CPDEMO1",
    "VETERAN3",
    "EMPLOY1",
    "CHILDREN",
    "INCOME2",
    "WEIGHT2",
    "HEIGHT3",
    "PREGNANT",
    "QLACTLM2",
    "USEEQUIP",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "SMOKDAY2",
    "STOPSMK2",
    "LASTSMK2",
    "USENOW3",
    "ALCDAY5",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXERANY2",
    "EXRACT11",
    "EXEROFT1",
    "EXERHMM1",
    "STRENGTH",
    "LMTJOIN3",
    "ARTHDIS2",
    "JOINPAIN",
    "FLUSHOT6",
    "FLSHTMY2",
    "PNEUVAC3",
    "HIVTST6",
    "HIVTSTD3",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "BLDSUGAR",
    "FEETCHK2",
    "DOCTDIAB",
    "CHKHEMO3",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVPRB1",
    "VINOCRE2",
    "CIMEMLOS",
    "WTCHSALT",
    "LONGWTCH",
    "DRADVISE",
    "ASTHMAGE",
    "ASATTACK",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTTODAY",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "SHINGLE2",
    "HADMAM",
    "HOWLONG",
    "HADPAP2",
    "LASTPAP2",
    "HPVTEST",
    "HPLSTTST",
    "HADHYST2",
    "PROFEXAM",
    "LENGEXAM",
    "BLDSTOOL",
    "HADSIGM3",
    "LASTSIG3",
    "PCPSARE1",
    "PSATEST1",
    "PSATIME",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTWRK1",
    "SCNTLWK1",
    "SXORIENT",
    "TRNSGNDR",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "MISTMNT",
    "ADANXEV",
    "QSTLANG",
    "MSCODE",
    "_CHISPNC",
    "_DUALUSE",
    "_RFHLTH",
    "_RFHYPE5",
    "_PRACE1",
    "_MRACE1",
    "_RACE",
    "_AGEG5YR",
    "_BMI5CAT",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_SMOKER3",
    "_RFSMOK3",
    "DRNKANY5",
    "DROCDY3_",
    "_RFBING5",
    "_DRNKWEK",
    "_RFDRHV5",
    "_FRTLT1",
    "_VEGLT1",
    "_TOTINDA",
    "ACTIN11_",
    "ACTIN21_",
    "_MINAC21",
    "_PACAT1",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
]

## 1. Load the training data into feature matrix and class labels
x_train, x_train_head, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(
    "data"
)

## 2. Convert labels from {-1, 1} to {0, 1}
y_train_p = helpers.process_labels(y_train)

## 3. Balance the data
x_train, y_train_p = helpers.balance_data(x_train, y_train)
print(f"dataset balanced with {x_train.shape[0]} samples")

## 4. Filter the features:
# First filter
x_train_f1, x_test_f1 = helpers.first_filter(x_train, x_train_head, x_test, filter1)
print(f"applied first filter, remaining features: {x_train_f1.shape[1]}")

# Replace NaN values with the mean of the column:
x_train_f1, x_test_f1 = helpers.replace_nan_with_median(x_train_f1, x_test_f1)

# Second filter
x_train_f2, x_test_f2 = helpers.second_filter(x_train_f1, x_test_f1, 1)
print(f"applied second filter, remaining features: {x_train_f2.shape[1]}")

## 5. Encode the categorical features and standardize the data
# x_train_p, x_test_p = helpers.process_features(x_train_f2, x_test_f2, 33)
# print(f'features processed, remaining features: {x_train_p.shape[1]}')

## 6. Train the model:
# Weights from least squares
# weights, _ = implementations.least_squares(y_train, x_train_f2)

# Initial weights from ridge regression
# weights, rmse = helpers.train_model(y_train_p, x_train_p, 2, 1, implementations.ridge_regression,
#                                     np.logspace(-4, 0, 5))
# print(f'using best w from ridge_regression with rmse {str(rmse)} as initial weights')

# Final weights from logistic regression
x_train_p = x_train_f2
x_test_p = x_test_f2
weights, loss = helpers.train_model(
    y_train_p,
    x_train_p,
    2,
    1,
    implementations.logistic_regression,
    helpers.calculate_nll,
    np.logspace(-4, 0, 5),
    initial_w=np.zeros(x_train_p.shape[1]),
    # initial_w=(np.random.random(x_train_p.shape[1]) - 0.5)*10,
    max_iters=100,
)
# weights, loss = implementations.reg_logistic_regression(y_train_p,
#                                         x_train_p,
#                                         lambda_=1e-5,
#                                         initial_w=weights,
#                                         max_iters=100,
#                                         gamma=1e-10)
print(f"final weights from logistic regression obtained with loss {loss}")

### 7. Make predictions:
y_pred = helpers.make_predictions_logistic_regression(weights, x_test_p)

# Store the predictions in a submission_file.csv in CSV format without index_label
helpers.create_csv_submission(test_ids, y_pred, "submission.csv")
