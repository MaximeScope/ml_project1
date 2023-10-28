import numpy as np
import helpers
import random


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: int
        gamma: float
    Returns:
        w: shape=(D, 1)
        loss: scalar
    """
    threshold = 1e-8
    w = initial_w
    loss = 0
    n_iter = 0
    prev_loss = np.inf

    for n_iter in range(max_iters):
        # Compute gradient and loss
        loss = helpers.compute_mse(y, tx, w)
        gradient = helpers.compute_grad(y, tx, w)

        # Update w by loss
        w = w - gamma * gradient
        # converge criterion
        if (prev_loss != np.inf) and np.abs(prev_loss - loss) < threshold:
            print(f"Converged with mse loss {loss} at iteration {iter}.")
            break
        prev_loss = loss
    if n_iter == max_iters - 1:
        print(f"Warning: reached max iterations {max_iters}.")

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w
    loss = 0
    threshold = 1e-8
    n_iter = 0
    prev_loss = np.inf

    for n_iter in range(max_iters):
        # Get random data point
        rand_index = random.randint(0, y.shape[0] - 1)
        y_rand = np.array([y[rand_index]])
        tx_rand = np.array([tx[rand_index]])

        # Compute gradient and loss
        loss = helpers.compute_mse(y_rand, tx_rand, w)
        gradient = helpers.compute_grad(y_rand, tx_rand, w)

        # Update w by loss
        w = w - gamma * gradient

        # converge criterion
        if (prev_loss != np.inf) and np.abs(prev_loss - loss) < threshold:
            print(f"Converged with mse loss {loss} at iteration {iter}.")
            break
        prev_loss = loss
    if n_iter == max_iters - 1:
        print(f"Warning: reached max iterations {max_iters}.")

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Using a linear algroithm solver following the formula for ridge regression
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    loss = helpers.compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Implementation of ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # Using a linear algroithm solver following the formula for ridge regression
    w = np.linalg.solve(
        np.transpose(tx).dot(tx) + 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1]),
        np.transpose(tx).dot(y),
    )
    loss = helpers.compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: int
        gamma: float
    """
    # init parameters
    threshold = 1e-4
    prev_loss = np.inf
    loss = 0
    w = initial_w
    iter = 0

    # start the logistic regression
    for iter in range(max_iters):
        # get loss, grad and update w.
        loss = helpers.calculate_nll(y, tx, w)
        grad = helpers.calculate_grad_nll(y, tx, w)
        w = w - gamma * grad
        # converge criterion
        if (prev_loss != np.inf) and np.abs(prev_loss - loss) < threshold:
            print(f"Converged with nnl loss {loss} at iteration {iter}.")
            break
        prev_loss = loss
    if iter == max_iters - 1:
        print(f"Warning: reached max iterations {max_iters}.")
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: shape=(D, 1)
        max_iters: int
        gamma: float
    """
    # init parameters
    threshold = 1e-4
    prev_loss = np.inf
    loss = 0
    iter = 0
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss, grad and update w.
        loss = helpers.calculate_nll(y, tx, w) + lambda_ * np.linalg.norm(w, 2) ** 2
        grad = helpers.calculate_grad_nll(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
        # converge criterion
        if (prev_loss != np.inf) and np.abs(prev_loss - loss) < threshold:
            print(f"Converged with nnl loss {loss} at iteration {iter}.")
            break
        prev_loss = loss
    if iter == max_iters - 1:
        print(f"Warning: reached max iterations {max_iters}.")
    return w, loss
