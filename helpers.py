"""Some helper functions for project 1."""
import csv
import numpy as np
import os
import implementations


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    x_train_head = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", max_rows=1, dtype=str
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )
    if sub_sample:
        y_train = np.genfromtxt(
            os.path.join(data_path, "y_train.csv"),
            delimiter=",",
            skip_header=1,
            dtype=int,
            usecols=1,
            max_rows=10000,
        )
        x_train = np.genfromtxt(
            os.path.join(data_path, "x_train.csv"),
            delimiter=",",
            skip_header=1,
            max_rows=10000,
        )
    else:
        y_train = np.genfromtxt(
            os.path.join(data_path, "y_train.csv"),
            delimiter=",",
            skip_header=1,
            dtype=int,
            usecols=1,
        )
        x_train = np.genfromtxt(
            os.path.join(data_path, "x_train.csv"),
            delimiter=",",
            skip_header=1,
        )
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_train_head = x_train_head[1:]
    x_test = x_test[:, 1:]

    return x_train, x_train_head, x_test, y_train, train_ids, test_ids


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed



    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def calculate_fscore(y, x, w, pred_func):
    pred_y = process_labels(pred_func(w, x))
    f_vec = y + 2 * pred_y
    tp = np.count_nonzero(f_vec == 3)
    fp = np.count_nonzero(f_vec == 2)
    fn = np.count_nonzero(f_vec == 1)
    print(f_vec)
    return tp / (tp + (fp + fn) / 2)


def cross_validation(
    y, x, k_indices, k, function, pred_func, param, initial_w=None, max_iters=None
):
    test_x = np.array([x[i] for i in k_indices[k]])
    test_y = np.array([y[i] for i in k_indices[k]])
    train_x = np.array([x[i] for i in range(len(x)) if i not in k_indices[k]])
    train_y = np.array([y[i] for i in range(len(y)) if i not in k_indices[k]])

    train_x, train_y = balance_data(train_x, train_y)

    if initial_w is None:
        w, _ = function(train_y, train_x, param)
    else:
        w, _ = function(train_y, train_x, initial_w, max_iters, param)

    f_score = calculate_fscore(test_y, test_x, w, pred_func)
    return f_score, w


def train_model(
    y, x, k_fold, seed, function, pred_func, params, initial_w=None, max_iters=None
):
    k_indices = build_k_indices(y, k_fold, seed)

    best_fscore = 0
    best_w = np.zeros(x.shape[1])
    print(f"Checking params {params}")
    for param in params:
        fscore_sum = 0
        w_sum = np.zeros(x.shape[1])
        for k in range(k_fold):
            f_score, w = cross_validation(
                y, x, k_indices, k, function, pred_func, param, initial_w, max_iters
            )
            fscore_sum += f_score
            w_sum += w
        curr_fscore = fscore_sum / k_fold
        print(f"Got F score {curr_fscore} for param {param}")
        if curr_fscore > best_fscore:
            best_fscore = curr_fscore
            best_w = w_sum / k_fold

    return best_w, best_fscore


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return 1 / (2 * len(y)) * np.sum((y - tx @ w) ** 2)


def compute_grad(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return -1 / len(y) * tx.T @ (y - tx @ w)


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """

    return 1 / (1 + np.exp(-t))


def calculate_nll(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    return (
        -1
        / len(y)
        * np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    )


def calculate_grad_nll(y, tx, w):
    """compute the gradient of negative log-likelihood loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """

    return 1 / len(y) * tx.T @ (sigmoid(tx @ w) - y)


def first_filter(x_train, x_train_head, x_test, filter):
    indexes_to_delete = []

    for col_index in range(len(list(x_train_head))):
        if list(x_train_head)[col_index] not in filter:
            indexes_to_delete.append(col_index)

    x_train_f1 = np.delete(x_train, indexes_to_delete, axis=1)
    x_test_f1 = np.delete(x_test, indexes_to_delete, axis=1)

    return x_train_f1, x_test_f1


def make_predictions_linear_regression(weights, x_test):
    y_pred = x_test.dot(weights)

    # Transform the predictions with values from -1 to 1
    y_pred_norm = 2 * (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()) - 1

    y_pred_norm[y_pred_norm > 0] = 1
    y_pred_norm[y_pred_norm <= 0] = -1

    return y_pred_norm


def make_predictions_logistic_regression(weights, x_test):
    y_pred = sigmoid(x_test.dot(weights))
    y_pred[y_pred <= 0.5] = -1
    y_pred[y_pred > 0.5] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def replace_nan_with_median(x_train, x_test):
    """
    Replace NaN values with the median of the column
    Args:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (N,D), D is the number of features.
    Returns:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (N,D), D is the number of features.
    """

    for col in range(x_train.shape[1]):
        nan_indices_train = np.isnan(x_train[:, col])
        nan_indices_test = np.isnan(x_test[:, col])
        col_median_train = np.nanmedian(x_train[:, col])
        x_train[nan_indices_train, col] = col_median_train
        x_test[nan_indices_test, col] = col_median_train
    return x_train, x_test


def second_filter(x_train, x_test, tol=1e-3):
    """
    Filter the features that are proportional to each other
    Args:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (N,D), D is the number of features.
        tol: tolerance for the allclose function
    Returns:
        x_train: numpy array of shape (N,D), D is the number of features.
        x_test: numpy array of shape (N,D), D is the number of features.
    """
    cols_filtered = [0]
    for col1 in range(1, x_train.shape[1]):
        is_prop = False
        for col2 in cols_filtered:
            if np.allclose(x_train[:, col1], x_train[:, col2], rtol=tol):
                is_prop = True
                break
        if not is_prop:
            cols_filtered.append(col1)
    return x_train[:, cols_filtered], x_test[:, cols_filtered]


def process_labels(y):
    y[y == -1] = 0
    y[y == 1] = 1
    return y


def onehot_encode_col(x_trainj, x_testj):
    unique = np.unique(x_trainj)
    x_trainj_onehot = np.zeros((len(x_trainj), len(unique)))
    x_testj_onehot = np.zeros((len(x_testj), len(unique)))

    for j in range(len(x_trainj)):
        x_trainj_onehot[j, np.where(unique == x_trainj[j])] = 1

    for j in range(len(x_testj)):
        x_testj_onehot[j, np.where(unique == x_testj[j])] = 1

    return x_trainj_onehot, x_testj_onehot


def standardize_col(x_trainj, x_testj):
    mean = np.mean(x_trainj)
    std = np.std(x_trainj)
    x_trainj = (x_trainj - mean) / std + 0.5
    x_testj = (x_testj - mean) / std + 0.5

    return x_trainj.reshape(len(x_trainj), -1), x_testj.reshape(len(x_testj), -1)


def process_features(x_train, x_test, onehot_thresh=100):
    x_train_processed = np.zeros((x_train.shape[0], 0))
    x_test_processed = np.zeros((x_test.shape[0], 0))

    for j in range(x_train.shape[1]):
        num_uniquej = len(np.unique(x_train[:, j]))

        # onehot encode if number of unique values is less than onehot_thresh
        if num_uniquej <= onehot_thresh:
            x_trainj_onehot, x_testj_onehot = onehot_encode_col(
                x_train[:, j], x_test[:, j]
            )
            x_train_processed = np.hstack((x_train_processed, x_trainj_onehot))
            x_test_processed = np.hstack((x_test_processed, x_testj_onehot))
            print(
                f"Feature index {j} has {num_uniquej} unique values --> onehot encoding"
            )

        else:
            # standardize if number of unique values is greater than onehot_thresh
            x_trainj_standardized, x_testj_standardized = standardize_col(
                x_train[:, j], x_test[:, j]
            )
            x_train_processed = np.hstack((x_train_processed, x_trainj_standardized))
            x_test_processed = np.hstack((x_test_processed, x_testj_standardized))
            print(
                f"Feature index {j} has {num_uniquej} > {onehot_thresh} unique values --> standardizing"
            )

    # Add bias term
    x_train_processed = np.hstack(
        (np.ones((x_train_processed.shape[0], 1)), x_train_processed)
    )
    x_test_processed = np.hstack(
        (np.ones((x_test_processed.shape[0], 1)), x_test_processed)
    )

    return x_train_processed, x_test_processed


def shuffle_data(x, y):
    """Shuffle the data"""
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    return x[shuffle_indices], y[shuffle_indices]


def balance_data(x_train, y_train):
    """
    Balance the data such that there are equal number of y_train entries = 0 and y_train entries = 1
    Args:
        x_train: input data
        y_train: output data (0 or 1)
    Returns:
        x_train_b: balanced input data
        y_train_b: balanced output data
    """
    # x_train, y_train = shuffle_data(x_train, y_train)
    # Count number of y_train entries = 1
    tot_ytrain1 = len(y_train[y_train == 1])

    # Create data subset such that there are equal number of y_train entries = 0 and y_train entries = 1
    x_train_b = []
    y_train_b = []
    cnt_ytrain1 = 0
    cnt_ytrain0 = 0
    for i in range(len(y_train)):
        if y_train[i] == 1:
            x_train_b.append(x_train[i])
            y_train_b.append(y_train[i])
            cnt_ytrain1 += 1
        elif y_train[i] == 0:
            if cnt_ytrain0 < tot_ytrain1:
                x_train_b.append(x_train[i])
                y_train_b.append(y_train[i])
                cnt_ytrain0 += 1

    x_train_b = np.array(x_train_b)
    y_train_b = np.array(y_train_b)
    # Shuffle the data
    x_train_b, y_train_b = shuffle_data(x_train_b, y_train_b)

    return x_train_b, y_train_b
