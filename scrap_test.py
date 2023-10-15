###############################################################################################
##### Don't worry about this code, it's just here to help me understanding numpy matrixes #####
###############################################################################################

import numpy as np

# Your original NumPy matrix
my_matrix = np.array([[1, np.nan, 3],
                      [np.nan, 5, 6],
                      [7, 8, 9]])

# Define an array of column indices to delete
column_indices = [0, 2]  # For example, to delete the first and third columns (0-based index)

# Use NumPy to delete the specified columns
# my_matrix = np.delete(my_matrix, column_indices, axis=1)

# Now, my_matrix contains the original matrix with the specified columns removed

print('--------------------------------------')
for col_index in range(my_matrix.shape[1]):
    # Find the indices of NaN values in the current column
    nan_indices = np.isnan(my_matrix[:, col_index])
    print("NaN indices:")
    print(nan_indices)
    # Calculate the mean of the current column, ignoring NaN values
    col_mean = np.nanmean(my_matrix[:, col_index])
    print("Col Mean:")
    print(col_mean)
    # Replace NaN values in the current column with the column mean
    my_matrix[nan_indices, col_index] = col_mean
print('--------------------------------------')
print(my_matrix)


### 1. Load the training data into feature matrix, class labels, and event ids:
# x_train = helpers.load_x('data/x_train.csv')
# y_train = helpers.load_y('data/y_train.csv')
x_train_head = np.genfromtxt('data/x_train.csv', delimiter=",", dtype=str, max_rows=1)
x_train = np.genfromtxt('data/x_train.csv', delimiter=",", skip_header=1)
y_train = np.genfromtxt('data/y_train.csv', skip_header=1)

### 2. Filter the features:
# First filter:
filter1 = ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2', 'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'ASTHMA3', 'CHCSCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY', 'DIABETE3', 'DIABAGE2', 'SEX', 'MARITAL', 'EDUCA', 'RENTHOM1', 'NUMHHOL2', 'CPDEMO1', 'VETERAN3', 'EMPLOY1', 'CHILDREN', 'INCOME2', 'WEIGHT2', 'HEIGHT3', 'PREGNANT', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'SMOKE100', 'SMOKDAY2', 'STOPSMK2', 'LASTSMK2', 'USENOW3', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'MAXDRNKS', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'EXERANY2', 'EXRACT11', 'EXEROFT1', 'EXERHMM1', 'STRENGTH', 'LMTJOIN3', 'ARTHDIS2', 'JOINPAIN', 'FLUSHOT6', 'FLSHTMY2', 'PNEUVAC3', 'HIVTST6', 'HIVTSTD3', 'PDIABTST', 'PREDIAB1', 'INSULIN', 'BLDSUGAR', 'FEETCHK2', 'DOCTDIAB', 'CHKHEMO3', 'DIABEYE', 'DIABEDU', 'CAREGIV1', 'CRGVREL1', 'CRGVPRB1', 'VINOCRE2', 'CIMEMLOS', 'WTCHSALT', 'LONGWTCH', 'DRADVISE', 'ASTHMAGE', 'ASATTACK', 'ASDRVIST', 'ASRCHKUP', 'ASACTLIM', 'ASYMPTOM', 'ASNOSLEP', 'ASTHMED3', 'ASINHALR', 'HAREHAB1', 'STREHAB1', 'CVDASPRN', 'ASPUNSAF', 'RLIVPAIN', 'RDUCHART', 'RDUCSTRK', 'ARTTODAY', 'ARTHWGT', 'ARTHEXER', 'ARTHEDU', 'TETANUS', 'HPVADVC2', 'SHINGLE2', 'HADMAM', 'HOWLONG', 'HADPAP2', 'LASTPAP2', 'HPVTEST', 'HPLSTTST', 'HADHYST2', 'PROFEXAM', 'LENGEXAM', 'BLDSTOOL', 'HADSIGM3', 'LASTSIG3', 'PCPSARE1', 'PSATEST1', 'PSATIME', 'SCNTMNY1', 'SCNTMEL1', 'SCNTWRK1', 'SCNTLWK1', 'SXORIENT', 'TRNSGNDR', 'EMTSUPRT', 'LSATISFY', 'ADPLEASR', 'ADDOWN', 'ADSLEEP', 'ADENERGY', 'ADEAT1', 'ADFAIL', 'ADTHINK', 'ADMOVE', 'MISTMNT', 'ADANXEV', 'QSTLANG', 'MSCODE', '_CHISPNC', '_DUALUSE', '_RFHLTH', '_RFHYPE5', '_PRACE1', '_MRACE1', '_RACE', '_AGEG5YR', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG', '_SMOKER3', '_RFSMOK3', 'DRNKANY5', 'DROCDY3_', '_RFBING5', '_DRNKWEK', '_RFDRHV5', '_FRTLT1', '_VEGLT1', '_TOTINDA', 'ACTIN11_', 'ACTIN21_', '_MINAC21', '_PACAT1', '_PAINDX1', '_PA150R2', '_PA300R2', '_PA30021', '_PASTRNG', '_PAREC1', '_LMTACT1', '_LMTWRK1', '_LMTSCL1', '_FLSHOT6', '_PNEUMO2', '_AIDTST3']
indexes_to_delete = []
for col_index in range(len(list(x_train_head))):
    if list(x_train_head)[col_index] in filter1:
        indexes_to_delete.append(col_index)
x_train_f1 = np.delete(x_train, indexes_to_delete, axis=1)
# print(x_train.shape)
# print(len(filter1))
# print(x_train_f1.shape)
# Second filter:
x_train_f2 = x_train_f1 # To-do with correlation

### 3 Train the model using least squares:
# Replace NaN values with the mean of the column:
for col_index in range(x_train_f2.shape[1]):
    # Find the indices of NaN values in the current column
    nan_indices = np.isnan(x_train_f2[:, col_index])
    # Calculate the mean of the current column, ignoring NaN values
    col_mean = np.nanmean(x_train_f2[:, col_index])
    # Replace NaN values in the current column with the column mean
    x_train_f2[nan_indices, col_index] = col_mean
# Generate the weights and the mse:

# Assuming you have your data in X and y

# List of alpha values to try
alphas = [0.1, 1.0, 10.0]

# Split the data into training and testing sets (or use cross-validation)
n_samples, n_features = x_train_f2.shape
n_train = int(0.8 * n_samples)  # 80% of the data for training

x_tr, y_tr = x_train_f2[:n_train], y_train[:n_train]
x_test, y_test = x_train_f2[n_train:], y_train[n_train:]

best_alpha = None
best_score = -np.inf
best_weights = None

for alpha in alphas:
    # Compute the Ridge regression weights using the closed-form solution
    identity_matrix = np.identity(n_features)
    weights = np.linalg.solve(x_tr.T @ x_tr + alpha * identity_matrix, x_tr.T @ y_tr)
    
    # Evaluate the model on the test data
    y_predict = x_test @ weights
    score = np.mean((y_predict - y_test) ** 2)  # Mean squared error as the score
    
    # Check if this alpha gives a better score
    if score > best_score:
        best_score = score
        best_alpha = alpha
        best_weights = weights