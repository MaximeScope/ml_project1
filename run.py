import numpy as np
import helpers
import implementations

# Setting random seed for the shuffling of the data points
np.random.seed(42)

# Manually selected list of features to delete:
# Ignore black formatting
# fmt: off
filter1 = ["GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH", "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4",
           "BPMEDS", "BPMEDS", "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3", "ASTHNOW", "ASTHMA3",
           "CHCSCNCR", "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY", "DIABETE3", "DIABAGE2", "SEX", "MARITAL", "EDUCA",
           "RENTHOM1", "NUMHHOL2", "CPDEMO1", "VETERAN3", "EMPLOY1", "CHILDREN", "INCOME2", "WEIGHT2", "HEIGHT3",
           "PREGNANT", "QLACTLM2", "USEEQUIP", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2",
           "STOPSMK2", "LASTSMK2", "USENOW3", "ALCDAY5", "AVEDRNK2", "DRNK3GE5", "MAXDRNKS", "FRUITJU1", "FRUIT1",
           "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1", "EXERANY2", "EXRACT11", "EXEROFT1", "EXERHMM1", "STRENGTH",
           "LMTJOIN3", "ARTHDIS2", "JOINPAIN", "FLUSHOT6", "FLSHTMY2", "PNEUVAC3", "HIVTST6", "HIVTSTD3", "PDIABTST",
           "PREDIAB1", "INSULIN", "BLDSUGAR", "FEETCHK2", "DOCTDIAB", "CHKHEMO3", "DIABEYE", "DIABEDU", "CAREGIV1",
           "CRGVREL1", "CRGVPRB1", "VINOCRE2", "CIMEMLOS", "WTCHSALT", "LONGWTCH", "DRADVISE", "ASTHMAGE", "ASATTACK",
           "ASDRVIST", "ASRCHKUP", "ASACTLIM", "ASYMPTOM", "ASNOSLEP", "ASTHMED3", "ASINHALR", "HAREHAB1", "STREHAB1",
           "CVDASPRN", "ASPUNSAF", "RLIVPAIN", "RDUCHART", "RDUCSTRK", "ARTTODAY", "ARTHWGT", "ARTHEXER", "ARTHEDU",
           "TETANUS", "HPVADVC2", "SHINGLE2", "HADMAM", "HOWLONG", "HADPAP2", "LASTPAP2", "HPVTEST", "HPLSTTST",
           "HADHYST2", "PROFEXAM", "LENGEXAM", "BLDSTOOL", "HADSIGM3", "LASTSIG3", "PCPSARE1", "PSATEST1", "PSATIME",
           "SCNTMNY1", "SCNTMEL1", "SCNTWRK1", "SCNTLWK1", "SXORIENT", "TRNSGNDR", "EMTSUPRT", "LSATISFY", "ADPLEASR",
           "ADDOWN", "ADSLEEP", "ADENERGY", "ADEAT1", "ADFAIL", "ADTHINK", "ADMOVE", "MISTMNT", "ADANXEV", "QSTLANG",
           "MSCODE", "_CHISPNC", "_DUALUSE", "_RFHLTH", "_RFHYPE5", "_PRACE1", "_MRACE1", "_RACE", "_AGEG5YR",
           "_BMI5CAT", "_CHLDCNT", "_EDUCAG", "_INCOMG", "_SMOKER3", "_RFSMOK3", "DRNKANY5", "DROCDY3_", "_RFBING5",
           "_DRNKWEK", "_RFDRHV5", "_FRTLT1", "_VEGLT1", "_TOTINDA", "ACTIN11_", "ACTIN21_", "_MINAC21", "_PACAT1",
           "_PAINDX1", "_PA150R2", "_PA300R2", "_PA30021", "_PASTRNG", "_PAREC1", "_LMTACT1", "_LMTWRK1", "_LMTSCL1",
           "_FLSHOT6", "_PNEUMO2", "_AIDTST3"]
# fmt: on

## 1. Load the training data into feature matrix and class labels
x_train, x_train_head, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(
    "dataset_to_release"
)

## 2. Convert labels from {-1, 1} to {0, 1}
y_train_p = helpers.process_labels(y_train)

## 3. First filter:
x_train_f1, x_test_f1 = helpers.first_filter(x_train, x_train_head, x_test, filter1)

## 4. Replace NaN values with the mean of the column:
x_train_f1, x_test_f1 = helpers.replace_nan_with_median(x_train_f1, x_test_f1)

## 5. Second filter:
x_train_f2, x_test_f2 = helpers.second_filter(x_train_f1, x_test_f1, 1)

## 6. Encode the categorical features and standardize the data
x_train_p, x_test_p = helpers.process_features(x_train_f2, x_test_f2, 10)

## 7. Train the model:
weights, f_score = helpers.train_model(
    y_train_p,
    x_train_p,
    4,
    1,
    implementations.logistic_regression,
    helpers.make_predictions_logistic_regression,
    np.logspace(-0.226625, -0.226375, 5),
    initial_w=np.zeros(x_train_p.shape[1]),
    max_iters=100,
)

## 8. Make predictions:
y_pred = helpers.make_predictions_logistic_regression(weights, x_test_p)

## 9. Store the predictions in a submission_file.csv in CSV format without index_label
helpers.create_csv_submission(test_ids, y_pred, "submission.csv")
