# ----- 0. SetUp - Import Packages -----
import numpy as np
import matplotlib.pyplot as plt


# ----- 1.1 Read the Data -----

def load_data(filename):
    """
    Load the data into two arrays

    Args:
        filename (str): The path to the text file containing the data (x, y)
    
    Return:
        array contraining the values of x
        array contraining the values of y
    """

    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]

x, y = load_data('Source/data_chol_dias_pressure.txt')


# ----- 1.2 Define the Cost Function -----

# The cost function is defined as the sum of squared errors between the predicted values and the actual values.
def cost_function (a, b, x, y):
    """
    Compute the cost function for given a and b

    Args:
        a : The slope of the line
        b : The y-intercept of the line
        x : The x-values of the data points
        y : The y-values of the data points

    Returns:
        The cost function value
    """
    cost = 0
    for i in range(len(x)):
        cost += (a * x[i] + b - y[i]) ** 2
    return cost


# ----- 1.3 Gradient Descent Algorithm to find optimal a and b -----

# First we can expand the previous approximate derivative functions to vectorized form:
def approx_df_linear(params, f, h=0.00001):
    """
    A function that implements the approximate derivative
    
    Args:
        params: A list or array containing the parameters such as (a, b)
        f: The cost function to compute the cost for given parameters
        h: The step size for the finite difference approximation
    
    Returns:
        An array containing the approximate derivatives with respect to parmeters like (a, b)
    """
    params = np.array(params)
    n = len(params)
    df = np.zeros(n)

    for i in range(n):
        # Create a unit vector e_i with 1 at the i-th position and 0 elsewhere
        e_i = np.zeros(n)
        e_i[i] = 1
        # For the i-th parameter, compute the approximate derivative using finite difference
        df[i] = (f(params + h * e_i) - f(params)) / h

    return df

# As we learned matrix and the way to solve using np.linalg.norm, 
# we can implement the gradient descent algorithm with approximate derivatives as follows:
def GD_approx(f, approx_df, h, params0, alpha, epsilon, iter_max=1000):
    """
    A function that implements the gradient descent algorithm with approximate derivatives
    
    Args:
        f: The cost function to minimize
        approx_df: A function that computes the approximate derivative of f
        h: The step size for the finite difference approximation
        params0: Initial guess for the parameters, such as (a, b)
        alpha: Learning rate
        epsilon: Convergence threshold
        iter_max: Maximum number of iterations
    
    Returns:
        The optimal parameters such as (a, b) and the number of iterations taken to converge
    """
    iter = 0
    params_current = np.array(params0)

    while (iter < iter_max):
        params_next = params_current - alpha * approx_df(params_current, f, h)

        if np.linalg.norm(params_next - params_current) < epsilon:
            return params_next, iter
        else:
            params_current = params_next
            iter += 1

    return params_next, iter





# First, we still need to standardize the data for the non-linear model

x_non_lin, y_non_lin = load_data('Source/data_chol_dias_pressure_non_lin.txt')

x_non_lin_scaled = (x_non_lin - np.mean(x_non_lin)) / np.std(x_non_lin)
y_non_lin_scaled = (y_non_lin - np.mean(y_non_lin)) / np.std(y_non_lin)

x_non_lin_train = x_non_lin_scaled[:15]
y_non_lin_train = y_non_lin_scaled[:15]

x_non_lin_test = x_non_lin_scaled[15:20]
y_non_lin_test = y_non_lin_scaled[15:20]

# Define the cost function for the non-linear model
def cost_function_non_lin(params, x, y):
    a, b, c = params
    cost = 0
    for i in range(len(x)):
        cost += (a * x[i]**2 + b * x[i] + c - y[i]) ** 2
    return cost

f_non_lin = lambda params: cost_function_non_lin(params, x_non_lin_train, y_non_lin_train)

def approx_df_non_lin(params, f, h=0.00001):
    a, b, c = params
    df_da = (f([a+h, b, c]) - f(params)) / h
    df_db = (f([a, b+h, c]) - f(params)) / h
    df_dc = (f([a, b, c+h]) - f(params)) / h
    return np.array([df_da, df_db, df_dc])

def GD_approx_non_lin(f, approx_df, h, params0, alpha, epsilon, iter_max=1000):
    iter = 0
    params_current = np.array(params0)

    while (iter < iter_max):
        params_next = params_current - alpha * approx_df(params_current, f, h)

        if np.linalg.norm(params_next - params_current) < epsilon:
            return params_next, iter
        else:
            params_current = params_next
            iter += 1
    return params_next, iter

params0 = [0, 0, 0]
params_opt, iter_non_lin = GD_approx_non_lin(f_non_lin, approx_df_non_lin, h=0.00001, params0=params0, alpha=0.01, epsilon=0.001, iter_max=1000)
print(params_opt, iter_non_lin)
a_non_lin_opt, b_non_lin_opt, c_non_lin_opt = params_opt
print(a_non_lin_opt, b_non_lin_opt, c_non_lin_opt)

# Calculate the cost for the non-linear model on the test set and the traning set
train_cost_non_lin = cost_function_non_lin(params_opt, x_non_lin_train, y_non_lin_train)
print(train_cost_non_lin)
test_cost_non_lin = cost_function_non_lin(params_opt, x_non_lin_test, y_non_lin_test)
print(test_cost_non_lin)


# Calculate the cost for the linear model on the test set
test_cost_lin = cost_function(a_original, b_original, x_non_lin_test, y_non_lin_test)
print(test_cost_lin)

# Transform the parameters back to the original scale
a_non_lin_original = a_non_lin_opt * (np.std(y_non_lin) / (np.std(x_non_lin)**2))
b_non_lin_original = b_non_lin_opt * (np.std(y_non_lin) / np.std(x_non_lin))
c_non_lin_original = c_non_lin_opt * np.std(y_non_lin) + np.mean(y_non_lin) - a_non_lin_original * np.mean(x_non_lin)**2 - b_non_lin_original * np.mean(x_non_lin)
print(a_non_lin_original, b_non_lin_original, c_non_lin_original)

# Update the plot to include the non-linear model
plt.scatter(x_non_lin, y_non_lin)
x_non_lin_sorted = np.sort(x_non_lin)
y_non_lin_pred = a_non_lin_original * x_non_lin_sorted**2 + b_non_lin_original * x_non_lin_sorted + c_non_lin_original
plt.plot(x_non_lin_sorted, y_non_lin_pred, label='Optimal Non-linear Model')
plt.legend()