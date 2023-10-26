import numpy as np
import helpers
import implementations


### 1. Load the training data into feature matrix and class labels

x_train, x_train_head, x_test, y_train, train_ids, test_ids = helpers.load_csv_data("data")

### 2. Filter the features:

# First filter:

# Manually selected list of features to delete:
filter1 = ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'ASTHMA3', 'CHCSCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'DIABAGE2', 'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'NUMHHOL2', 'CPDEMO1', 'VETERAN3', 'EMPLOY1', 'CHILDREN', 'INCOME2', 'WEIGHT2', 'HEIGHT3', 'PREGNANT', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', 'USENOW3', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'STRENGTH', 'LMTJOIN3', 'ARTHDIS2', 'JOINPAIN', 'FLUSHOT6', 'FLSHTMY2', 'PNEUVAC3', 'HIVTST6', 'HIVTSTD3', 'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2', 'DOCTDIAB', 'CHKHEMO3', 'DIABEYE', 'DIABEDU', 'CAREGIV1', 'CRGVREL1', 'CRGVPRB1', 'VINOCRE2', 'CIMEMLOS', 'WTCHSALT', 'LONGWTCH', 'DRADVISE', 'ASTHMAGE', 'ASATTACK', 'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP', 'ASTHMED3', 'ASINHALR', 'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'ASPUNSAF', 'RLIVPAIN', 'RDUCHART', 'RDUCSTRK', 'ARTTODAY', 'ARTHWGT', 'ARTHEXER', 'ARTHEDU', 'TETANUS', 'HPVADVC2', 'SHINGLE2', 'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PROFEXAM', 'LENGEXAM', 'BLDSTOOL', 'HADSIGM3', 'LASTSIG3', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'SCNTMNY1', 'SCNTMEL1', 'SCNTWRK1', 'SCNTLWK1', 'SXORIENT', 'TRNSGNDR', 'EMTSUPRT', 'LSATISFY', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE', 'MISTMNT', 'ADANXEV', 'QSTLANG', 'MSCODE', '_CHISPNC', '_DUALUSE', '_RFHLTH', '_RFHYPE5', '_PRACE1', '_MRACE1', '_RACE', '_AGEG5YR', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWEK', '_RFDRHV5', '_FRTLT1', '_VEGLT1', '_TOTINDA', 'ACTIN11_', 'ACTIN21_', '_MINAC21', '_PACAT1', '_PAINDX1', '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_FLSHOT6', '_PNEUMO2', '_AIDTST3']
# filter2 = ['GENHLTH', 'CVDSTRK3', 'EXRACT11', 'POORHLTH', 'PHYSHLTH', '_AGEG5YR',
#        'CHCKIDNY', 'EMPLOY1', 'CHCCOPD1', '_LMTSCL1', 'DIABETE3', 'MENTHLTH',
#        '_RFHLTH', 'USEEQUIP', 'DIFFDRES']

x_train_f1, x_test = helpers.first_filter(x_train, x_train_head, x_test, filter1)

# Second filter:

# Replace NaN values with the mean of the column:
for col_index in range(x_train_f1.shape[1]):
    # Find the indices of NaN values in the current column of x_train_f1
    nan_indices_train_f1 = np.isnan(x_train_f1[:, col_index])
    # Find the indices of NaN values in the current column of x_test
    nan_indices_test = np.isnan(x_test[:, col_index])
    # Calculate the mean of the current column of x_train_f1, ignoring NaN values
    col_mean = np.nanmean(x_train_f1[:, col_index])
    # Replace NaN values in the current column of x_train_f1 with the column mean of x_train_f1
    x_train_f1[nan_indices_train_f1, col_index] = col_mean
    # Replace NaN values in the current column of x_test with the column mean of x_train_f1
    x_test[nan_indices_test, col_index] = col_mean

tolerance = 1e-8  # Tolerance for comparing scalar multiples (adjust as needed)

# Create a list to store column indices to keep
columns_to_keep = [0]  # Start with the first column, or any initial choice

# Iterate through the columns, starting from the second column
for col1 in range(1, x_train_f1.shape[1]):
    is_proportional = False  # Flag to check if the column is proportional to any of the columns to keep
    
    for col2 in columns_to_keep:
        if np.allclose(x_train_f1[:, col1], x_train_f1[:, col2], atol=tolerance):
            is_proportional = True
            # print(f"Column {col1} is proportional to Column {col2}. It will be deleted.")
            break  # The column is proportional, so break
    
    if not is_proportional:
        columns_to_keep.append(col1)

# Create a new filtered x_train with the selected columns
x_train_f2 = x_train_f1[:, columns_to_keep]
# 'x_train_f2' now contains the columns that are not proportional to each other.

# Filter x_test with the columns that we kept from the training data
x_test = x_test[:, columns_to_keep]

### 3 Train the model using least squares:

# Generate the weights and the mse:

# Weights from least squares
# weights, _ = implementations.least_squares(y_train, x_train_f2)

# Weights from ridge regression
weights, rmse = helpers.train_ridge_regression(y_train, x_train_f2, 2, np.logspace(-4, 0, 5), 1)
print("best w: " + str(w) + " with rmse " + str(rmse))

### 4. Make predictions:

y_pred_norm = helpers.make_predictions(weights, x_test)

# Store the predictions in a submission_file.csv in CSV format without index_label
helpers.create_csv_submission(test_ids, y_pred_norm, 'submission_file.csv')

### 5. Train model using ridge regression
w, rmse = helpers.train_ridge_regression(y_train, x_train_f2, 2, np.logspace(-4, 0, 5), 1)
print("best w: " + str(w) + " with rmse " + str(rmse))
