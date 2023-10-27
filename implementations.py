import numpy as np
import helpers
import random


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of GD
            gamma: a scalar denoting the stepsize

        Returns:
            losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
            ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
        """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # Compute gradient and loss
        loss = helpers.compute_mse(y, tx, w)
        gradient = helpers.compute_gradient(y, tx, w)

        # Update w by loss
        w = w - gamma * gradient

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
    for n_iter in range(max_iters):
        # Get random data point
        rand_index = random.randint(0, y.shape[0] - 1)
        y_rand = y[rand_index]
        tx_rand = tx[rand_index]

        # Compute gradient and loss
        loss = helpers.compute_mse(y_rand, tx_rand, w)
        gradient = helpers.compute_gradient(y_rand, tx_rand, w)

        # Update w by loss
        w = w - gamma * gradient

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

    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    return w, helpers.compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

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
    return np.linalg.solve(np.transpose(tx).dot(tx) + 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1]),
                           np.transpose(tx).dot(y))


def logistic_regression(y, tx, initial_w, gamma, max_iters=100, convergeance_thresh=1e-3):
    w = initial_w
    prev_loss = 99999
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = helpers.learning_by_newton_method(y, tx, w, gamma)
        print(f'Current iteration={iter}, loss={loss}')

        # convergeance check
        if (prev_loss - loss) < convergeance_thresh:
            print(f'Converged!')
            break
        else: 
            prev_loss = loss

    return w, loss


def reg_logistic_regression(y, tx, initial_w, gamma, lambda_, max_iters=100, convergeance_thresh=1e-3):
    w = initial_w
    prev_loss = 99999
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = helpers.learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        print(f'Current iteration={iter}, loss={loss}')

        # convergeance check
        if (prev_loss - loss) < convergeance_thresh:
            print(f'Converged!')
            break
        else: 
            prev_loss = loss
    return w, loss