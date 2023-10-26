"""Some helper functions for project 1."""
import csv
import numpy as np

import implementations

def load(data_path, sub_sample=False):
    headers = np.genfromtxt('data/x_train.csv', delimiter=",", dtype=str, max_rows=1)
    if sub_sample:
        data = np.genfromtxt('data/x_train.csv', delimiter=",", skip_header=1, max_rows=100)
    else:
        data = np.genfromtxt('data/x_train.csv', delimiter=",", skip_header=1)
    return data, headers

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


def ridge_regression_cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> ridge_regression_cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2)
    (0.019866645527597114, 0.33555914361295175)
    """
    test_x = np.array([x[i] for i in k_indices[k]])
    test_y = np.array([y[i] for i in k_indices[k]])
    train_x = np.array([x[i] for i in range(len(x)) if i not in k_indices[k]])
    train_y = np.array([y[i] for i in range(len(y)) if i not in k_indices[k]])

    w = implementations.ridge_regression(train_y, train_x, lambda_)

    loss_te = np.sqrt(np.mean((test_y - test_x.dot(w)) ** 2))
    return loss_te, w


def train_ridge_regression(y, x, k_fold, lambdas, seed):
    k_indices = build_k_indices(y, k_fold, seed)

    best_rmse = 99999
    best_w = np.zeros(x.shape[1])
    for lambda_ in lambdas:
        print("Checking lambda " + str(lambda_))
        loss_te_sum = 0
        w_sum = np.zeros(x.shape[1])
        for k in range(k_fold):
            loss_te, w = ridge_regression_cross_validation(y, x, k_indices, k, lambda_)
            loss_te_sum += loss_te
            w_sum += w

        curr_rmse = loss_te_sum / k_fold
        if curr_rmse < best_rmse:
            print("Got best w with lambda " + str(lambda_) + " with rmse " + str(curr_rmse))
            best_rmse = curr_rmse
            best_w = w_sum / k_fold

    return best_w, best_rmse


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean((y - tx.dot(w)) ** 2) / 2

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return -np.transpose(tx).dot(y - tx.dot(w)) / np.shape(y)[0]


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
    return np.exp(t) / (1 + np.exp(t))


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
    >>> round(calculate_nll(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sum = 0
    for i in range(y.shape[0]):
        sum += - y[i]*tx[i].dot(w) - np.log(1-sigmoid(tx[i].dot(w)))
    return sum[0] / y.shape[0]


def calculate_nll_gradient(y, tx, w):
    """compute the gradient of loss.

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
    >>> calculate_nll_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    return tx.transpose().dot(sigmoid(tx.dot(w)) - y) / y.shape[0]


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    s = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        sig = sigmoid(tx[i].dot(w))[0]
        s[i][i] = sig * (1 - sig)
    return tx.transpose().dot(s).dot(tx) / y.shape[0]


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step of Newton's method.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 0., 1., 1.]]
    >>> np.random.seed(0)
    >>> tx = np.random.rand(4, 3)
    >>> w = np.array([[0.1], [0.5], [0.5]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_newton_method(y, tx, w, gamma)
    >>> round(loss, 8)
    0.71692036
    >>> w
    array([[-1.31876014],
           [ 1.0590277 ],
           [ 0.80091466]])
    """
    loss, gradient, hessian = calculate_nll(y, tx, w), calculate_nll_gradient(y, tx, w), calculate_hessian(y, tx, w)

    w_diff = np.linalg.solve(hessian, gamma * gradient)

    return loss, w - w_diff


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    return calculate_nll(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2, calculate_nll_gradient(y, tx, w) + 2 * lambda_ * w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w


def create_csv_submission(y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1 in y_pred:
            writer.writerow({"Prediction": int(r1)})
