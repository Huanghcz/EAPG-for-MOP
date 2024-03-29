import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

"""
    the program for the accelerated proximal gradient method for non-smooth multi-objective problem
"""

# define objective functions and their gradient functions,proximal operator
def f1(x, mu):
    res = 0
    if np.any(mu > 0):
        res = mu * np.log(np.exp((x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1) / mu) + np.exp(
            (-x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1) / mu))
    return res

def grad_f1(x, mu):
    res = 0
    if np.any(mu > 0):
        a = x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1
        b = -x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1
        term1 = ((np.exp(a / mu) - np.exp(b / mu)) * (2 * x[0])) / (np.exp(a / mu) + np.exp(b / mu))
        term2 = ((2 * (x[0] - 1) + 1) * (np.exp(a / mu)) + (-2 * (x[0] - 1) + 1) * (np.exp(b / mu))) / (
                    np.exp(a / mu) + np.exp(b / mu))
        res = np.array([term1, term2])
    return res

def f2(x, mu):
    sum = np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    if np.any(sum > mu):
        res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        res = (x[0] ** 2 + x[1] ** 2 - 1) ** 2 / (2 * mu) + mu / 2
    return res

def grad_f2(x, mu):
    condition = np.any(abs(x[0] ** 2 + x[1] ** 2 - 1) > mu)

    if condition:
        df_dx1 = -3.5 * x[0] * abs(x[0] ** 2 + x[1] ** 2 - 1)
        df_dx2 = -3.5 * x[1] * abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        df_dx1 = (x[0] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu
        df_dx2 = (x[1] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu

    return np.array([df_dx1, df_dx2])


def g1(x):
    return 0


def g2(x):
    res = - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res


def f(x, mu):
    return np.array([f1(x, mu), f2(x, mu)])


def g(x):
    res = np.array([g1(x), g2(x)])
    return res


def jac_f(x, mu):
    term11 = grad_f1(x, mu)[0]
    term12 = grad_f1(x, mu)[1]
    term21 = grad_f2(x, mu)[0]
    term22 = grad_f2(x, mu)[1]
    res = np.array([[term11, term12], [term21, term22]])
    return res

# define the subproblem solver
def optimize_process(x0):
    k = 0
    iter_max = 100
    while k <= iter_max:
        xk_old = x0
        yk = xk_old
        mu_k = 1
        w = 0.5
        lr = 0.5
        beta = 3
        epsilon = 0.0001
        mu0 = 1
        sigma = 0.7
        lam = np.array([w, 1 - w])
        grad_f1_x = grad_f1(yk, mu_k)  # 1*2
        grad_f2_x = grad_f2(yk, mu_k)  # 1*2
        w = float(w)
        max_iter = 1000
        tol = 1e-6
        wsum_nabla_f_yk = w * grad_f1_x + (1 - w) * grad_f2_x  # 1*2
        y_minus_wsum_nabla_f_yk = yk - lr * wsum_nabla_f_yk  # 1*2
        F_xk_old_mu_k = np.array([f1(xk_old, mu_k), f2(xk_old, mu_k)]) + g(xk_old)  # 1*2

        def prox_wsum_g(x, w):
            return np.array([(x[0] + 1 - w) / (4 * (1 - w) + 1), x[1] / (4 * (1 - w) + 1)])

        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, lr*w)

        def fun(w):
            primal_value_minus_y = primal_value - y_minus_wsum_nabla_f_yk
            return np.inner(lam, g(primal_value)) + np.linalg.norm(primal_value_minus_y) / (2 * lr) - (
                    lr / 2) * np.linalg.norm(
                wsum_nabla_f_yk) ** 2 + w * (f1(yk, mu_k) - f1(xk_old, mu_k) - g(xk_old)[0]) + (1 - w) * (
                               f2(yk, mu_k) - f2(xk_old, mu_k) - g(xk_old)[1])

        res = minimize(
                fun,
                x0= 0.5,  # 初始猜测
                bounds = [(0, 1)],
                options={"maxiter": max_iter, "ftol": tol},
            )

        lam_star = np.array([res.x, 1 - res.x])
        wsum_nabla_f_yk_star = lam_star[0] * grad_f1_x+ lam_star[1] * grad_f2_x
        #wsum_nabla_f_yk_star = lam_star.flatten().dot(nabla_f_yk)
        y_minus_wsum_nabla_f_yk_star = yk - lr * wsum_nabla_f_yk_star
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,lr*res.x)
        if np.linalg.norm(primal_value_star - yk) >= epsilon:
            xk = primal_value_star
            mu_k = mu_k / ((k + beta - 1) * (np.log(k + beta - 1) ** sigma))
            gamma_k = (k - 1) / (k + beta - 1)
            yk_plus_1 = xk + gamma_k * (xk - xk_old)
            # Update f, g, f_yk, nabla_f_yk here as needed
            yk = yk_plus_1
            k += 1
    return xk

# define the primal objective funtion
def f11(x):
    res = np.maximum(x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1, -x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1)
    return res

def f22(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res


# general initial points
initial_points = np.random.uniform(-2,2,size=(50,2))

# save the optimal result
f11_values = []
f22_values = []

# save the initial result
f11_initial_values = []
f22_initial_values = []

# compute the result for all the initial points
for point in initial_points:
    optimized_result = optimize_process(point)
    f110 = f11(point)
    f220 = f22(point)
    f111 = f11(optimized_result)
    f221 = f22(optimized_result)

    # check
    if not (f111 > f110 and f221 > f220):
        f11_initial_values.append(f110)
        f22_initial_values.append(f220)
        f11_values.append(f111)
        f22_values.append(f221)

# plot the result
plt.figure(figsize=(10, 6))
plt.scatter(f11_initial_values, f22_initial_values, c='blue', marker='x', label='Initial Points')
plt.scatter(f11_values, f22_values, c='red', marker='o', label='Optimized Points')
plt.xlabel('f11')
plt.ylabel('f22')
plt.title('Pareto Front of Optimization Results')
plt.legend()

plt.xlim(-10, 10)
plt.ylim(-20, 20)

plt.savefig("pareto_front.png")
plt.show()