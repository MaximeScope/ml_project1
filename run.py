import numpy as np
import csv
import helpers
import implementations


### 1. Load the training data into feature matrix and class labels

# x_train = helpers.load_x('data/x_train.csv')
# y_train = helpers.load_y('data/y_train.csv')
x_train_head = np.genfromtxt('data/x_train.csv', delimiter=",", dtype=str, max_rows=1)
x_train = np.genfromtxt('data/x_train.csv', delimiter=",", skip_header=1)
y_train = np.genfromtxt('data/y_train.csv', skip_header=1)


### 2. Filter the features:

# First filter:

# Manually selected list of features to delete:
filter1 = ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'ASTHMA3', 'CHCSCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'DIABAGE2', 'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'NUMHHOL2', 'CPDEMO1', 'VETERAN3', 'EMPLOY1', 'CHILDREN', 'INCOME2', 'WEIGHT2', 'HEIGHT3', 'PREGNANT', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', 'USENOW3', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'STRENGTH', 'LMTJOIN3', 'ARTHDIS2', 'JOINPAIN', 'FLUSHOT6', 'FLSHTMY2', 'PNEUVAC3', 'HIVTST6', 'HIVTSTD3', 'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2', 'DOCTDIAB', 'CHKHEMO3', 'DIABEYE', 'DIABEDU', 'CAREGIV1', 'CRGVREL1', 'CRGVPRB1', 'VINOCRE2', 'CIMEMLOS', 'WTCHSALT', 'LONGWTCH', 'DRADVISE', 'ASTHMAGE', 'ASATTACK', 'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP', 'ASTHMED3', 'ASINHALR', 'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'ASPUNSAF', 'RLIVPAIN', 'RDUCHART', 'RDUCSTRK', 'ARTTODAY', 'ARTHWGT', 'ARTHEXER', 'ARTHEDU', 'TETANUS', 'HPVADVC2', 'SHINGLE2', 'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PROFEXAM', 'LENGEXAM', 'BLDSTOOL', 'HADSIGM3', 'LASTSIG3', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'SCNTMNY1', 'SCNTMEL1', 'SCNTWRK1', 'SCNTLWK1', 'SXORIENT', 'TRNSGNDR', 'EMTSUPRT', 'LSATISFY', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE', 'MISTMNT', 'ADANXEV', 'QSTLANG', 'MSCODE', '_CHISPNC', '_DUALUSE', '_RFHLTH', '_RFHYPE5', '_PRACE1', '_MRACE1', '_RACE', '_AGEG5YR', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWEK', '_RFDRHV5', '_FRTLT1', '_VEGLT1', '_TOTINDA', 'ACTIN11_', 'ACTIN21_', '_MINAC21', '_PACAT1', '_PAINDX1', '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_FLSHOT6', '_PNEUMO2', '_AIDTST3']
# filter2 = ['GENHLTH', 'CVDSTRK3', 'EXRACT11', 'POORHLTH', 'PHYSHLTH', '_AGEG5YR',
#        'CHCKIDNY', 'EMPLOY1', 'CHCCOPD1', '_LMTSCL1', 'DIABETE3', 'MENTHLTH',
#        '_RFHLTH', 'USEEQUIP', 'DIFFDRES']

indexes_to_delete = []

for col_index in range(len(list(x_train_head))):
    if list(x_train_head)[col_index] in filter1:
        indexes_to_delete.append(col_index)

x_train_f1 = np.delete(x_train, indexes_to_delete, axis=1)

# Second filter:

# Replace NaN values with the mean of the column:
for col_index in range(x_train_f1.shape[1]):
    # Find the indices of NaN values in the current column
    nan_indices = np.isnan(x_train_f1[:, col_index])
    # Calculate the mean of the current column, ignoring NaN values
    col_mean = np.nanmean(x_train_f1[:, col_index])
    # Replace NaN values in the current column with the column mean
    x_train_f1[nan_indices, col_index] = col_mean

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


### 3 Train the model using least squares:

# Generate the weights and the mse:

# weights = np.dot(np.linalg.inv(x_train_f2.T.dot(x_train_f2)), x_train_f2.T).dot(y_train)
weights, _ = implementations.least_squares(y_train, x_train_f2)

### 4. Make predictions:
# Use weights to predict which columns correlate the most with y_train
# print("Length of filter1: "+str(len(filter1)))
# print("Shape of x_train_f2: "+str(x_train_f2.shape))
# print("Shape of the weights: "+str(weights.shape))
y_pred = x_train_f2.dot(weights)
# Transform the predictions with values from 0 to 1
y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
# If the value is above 0.5, consider it to be 1 and otherwise 0
y_pred_norm[y_pred_norm > 0.5] = 1
y_pred_norm[y_pred_norm <= 0.5] = 0
# Store the predictions in a submission_file.csv in CSV format without index_label
helpers.create_csv_submission(y_pred_norm, 'submission_file.csv')