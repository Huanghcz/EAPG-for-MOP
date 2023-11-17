import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
"""
    this test use apgm to sovle problem :Crescent & Mifflin 2,
    which start point is (-1,-1),
    we get the optimal result is (0.1632,-0.3042),
    which is better than (0.17156, −0.72429) in paper[1],
    and better than (0.3939, −0.8529) in paper[2]
    [1] Mäkelä M M, Karmitsa N, Wilppu O. Multiobjective proximal bundle method for nonsmooth optimization[R]. TUCS, Technical Report, 2014.
    [2] Hoseini Monjezi N, Nobakhtian S. An inexact multiple proximal bundle algorithm for nonsmooth nonconvex multiobjective optimization problems[J]. 
        Annals of Operations Research, 2022, 311(2): 1123-1154.
"""

def f1(x, mu):
    a = x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    b = -x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    c = x[0] - 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    if mu > 0:
        res = mu * np.log(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu))
    return res


def grad_f1(x, mu):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    if mu > 0:
        res1 = (0.5 *(1+ 1/((x[0]+0.1)**2))/mu) * np.exp(0.5*a/mu)+0.5 *(-1+ 1/((x[0]+0.1)**2))/mu * np.exp(0.5*a/mu) +0.5 *(1 - 1/((x[0]+0.1)**2))/mu * np.exp(0.5*a/mu)
        diff1 = mu * ((res1)/(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu)))
        res2 = (2*x[1]/mu) * np.exp(0.5*a/mu) + (2*x[1]/mu) * np.exp(0.5*b/mu) + (2*x[1]/mu) * np.exp(0.5*c/mu)
        diff2 = mu * ((res2)/(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu)))
        res = np.array([diff1,diff2])
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




def optimize_process(x0):
    k = 0
    iter_max = 100
    f11_prev = f11(x0)
    f22_prev = f22(x0)
    xk_old = x0
    yk = xk_old
    mu_k = 4/5
    w = 0.5
    lr = 1e-9
    beta = 4
    epsilon = 0.0001
    mu0 = 1
    sigma = 3/4
    lam = np.array([w, 1 - w])
    grad_f1_x = grad_f1(yk, mu_k)  # 1*2
    grad_f2_x = grad_f2(yk, mu_k)  # 1*2
    w = float(w)
    max_iter = 1000
    tol = 1e-12
    while k <= iter_max:
        wsum_nabla_f_yk = w * grad_f1_x + (1 - w) * grad_f2_x  # 1*2
        y_minus_wsum_nabla_f_yk = yk - lr * wsum_nabla_f_yk  # 1*2
        F_xk_old_mu_k = np.array([f1(xk_old, mu_k), f2(xk_old, mu_k)]) + g(xk_old)  # 1*2

        def prox_wsum_g(x, w):
            return np.array([(x[0] + 1 - w) / (4 * (1 - w) + 1), x[1] / (4 * (1 - w) + 1)])

        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, lr*mu_k*w)

        def fun(w):
            primal_value_minus_y = primal_value - y_minus_wsum_nabla_f_yk
            return -np.inner(lam, g(primal_value)) - np.linalg.norm(primal_value_minus_y) / (2 * lr) + (
                    lr / 2) * np.linalg.norm(
                wsum_nabla_f_yk) ** 2 + w * (f1(yk, mu_k) - f1(xk_old, mu_k) - g(xk_old)[0]) + (1 - w) * (
                               f2(yk, mu_k) - f2(xk_old, mu_k) - g(xk_old)[1])

        res = minimize(
                fun,
                x0= 0.5,
                bounds = [(0, 1)],
                options={"maxiter": max_iter, "ftol": tol},
            )

        lam_star = np.array([res.x, 1 - res.x])
        wsum_nabla_f_yk_star = lam_star[0] * grad_f1_x+ lam_star[1] * grad_f2_x
        #wsum_nabla_f_yk_star = lam_star.flatten().dot(nabla_f_yk)
        y_minus_wsum_nabla_f_yk_star = yk - lr * wsum_nabla_f_yk_star
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,lr*mu_k*res.x)
        primal_value_star_re = np.array([float(primal_value_star[0]), float(primal_value_star[1])])
        if np.linalg.norm(primal_value_star - yk,ord=np.infty) >= epsilon:
            xk = primal_value_star_re
            mu_k = mu_k / ((k + beta - 1) * (np.log(k + beta - 1) ** sigma))
            gamma_k = (k - 1) / (k + beta - 1)
            yk_plus_1 = xk + gamma_k * (xk - xk_old)
            # Update f, g, f_yk, nabla_f_yk here as needed
            yk = yk_plus_1
            k +=1
        else:
            print(f'符合判断准则，输出xk')
            xk = primal_value_star_re
            return xk

        f11_current = f11(xk)
        f22_current = f22(xk)
        if f11_prev -f11_current < 1e-22 and f22_prev-f22_current < 1e-22:
            print(f'迭代终止于第 {k} 次')
            break
        f11_prev = f11_current
        f22_prev = f22_current
    return xk

def f11(x):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    res = np.max([0.5 * a, 0.5 * b, 0.5 * c])
    return res


def f22(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res

# general initial points
np.random.seed(0)
initial_points = np.random.uniform(-2, 2, (100, 2))

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
# plt.scatter(f11_initial_values, f22_initial_values, c='blue', marker='x', label='Initial Points')
plt.scatter(f11_values, f22_values, c='red', marker='o', label='Optimized Points')
plt.xlabel('f11')
plt.ylabel('f22')
plt.title('Pareto Front of Optimization Results')
plt.legend()



plt.savefig("WF&Mifflin_pareto_front.png")
plt.show()