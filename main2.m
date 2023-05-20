function result = main2()
% 设置初始条件
m = 8;
n = 6;
A = rand(m, n);
b = rand(m, 1);
x_prev_prev = zeros(n, 1);
x_prev = zeros(n, 1);
x_current = zeros(n, 1);
y_current = zeros(n, 1);

mu_0 = 1;
mu_prev_prev = mu_0;
mu_prev = mu_0;
mu_current = mu_0;

t_current = 1;
epsilon = 1e-6;
alpha = 0.5;
sigma = 0.25;
L = 0.01;
beta_k = 0.01;
maxIter = 4000;
k = 1;
l = 0.01; % 添加这一行来定义 l
% 主循环
% Initialize array to store solutions
solutions = zeros(n, maxIter);
while (k <= maxIter && mu_current > 0.00001)
    if norm(x_current - y_current, Inf) < epsilon
        nabla_f_tilde = numerical_gradient(@(x) f_tilde(A, x, b, mu_prev, m), y_current);
        phi_acc_l = @(z) max(nabla_f_tilde'*(z - y_current) + 0.01*norm(z, 1) + f_tilde(A, y_current, b, mu_current, m) - F_tilde(A, x_prev, b, mu_prev, m)) + l/mu_current/2 * norm(z - y_current)^2;
        x_current = fminunc(phi_acc_l, x_current);
    end
   

    tau_k = L/mu_prev/4 + L*beta_k^2/mu_current/4;
    H_prev = F_tilde(A, x_prev, b, mu_prev, m) + tau_k * norm(x_prev - x_prev_prev)^2;
    H_current = F_tilde(A, x_current, b, mu_current, m) + tau_k * norm(x_current - x_prev)^2;

    if H_current + mu_prev - H_prev + mu_prev_prev <= -alpha * mu_current^2
        mu_next = mu_current;
    else
        mu_next = mu_0 / (k + 1)^sigma;
    end

    t_next = (1 + sqrt(1 + 4*mu_current/mu_next*t_current^2)) / 2;
    gamma_k = (t_current - 1) / t_next;

    y_next = x_current + gamma_k * (x_current - x_prev);

    % 更新变量
    x_prev_prev = x_prev;
    x_prev = x_current;
    x_current = y_next;

    mu_prev_prev = mu_prev;
    mu_prev = mu_current;
    mu_current = mu_next;

    t_current = t_next;

    k = k + 1;
    
    % Save the current solution
    solutions(:, k) = x_current;

    % Display the iteration number and the current solution
    fprintf('Iteration number: %d, Current solution: \n', k);
    disp(x_current);

    k = k + 1;
end
% 绘制解的变化情况
figure;
for i = 1:n
    subplot(n, 1, i);
    plot(1:20:k-1, solutions(i, 1:20:k-1), 'LineWidth', 2); % 每20次迭代显示一次
    title(sprintf('Convergence of x%d', i));
    xlabel('Iteration');
    ylabel(sprintf('x%d', i));
end
% Plot the convergence of the solutions
figure;
for i = 1:n
    plot(1:maxIter, solutions(i, :));
    hold on;
end
xlabel('Iteration number');
ylabel('Solution value');
title('Convergence of the solution');
legend('x1', 'x2', 'x3', 'x4', 'x5', 'x6'); % Adjust this according to the dimension of your problem
hold off;
end



function result = theta_tilde(s, mu)
    if abs(s) > mu
        result = abs(s);
    else
        result = s^2/(2*mu) + mu/2;
    end
end

function result = phi_tilde(s, mu)
    if abs(s) > mu
        result = max(s, 0);
    else
        result = (s + mu)^2 / (4 * mu);
    end
end

function result = f_tilde(A, x, b, mu, m)
    result = [0; 0];
    for i = 1:m
        result(1) = result(1) + theta_tilde(A(i,:)*x - b(i), mu);
        result(2) = result(2) + theta_tilde(phi_tilde(A(i,:)*x - b(i), mu), mu);
    end
end

function result = F_tilde(A, x, b, mu, m)
    g = 0.01 * norm(x, 1);
    result = f_tilde(A, x, b, mu, m) + [g; g];
end
function grad = numerical_gradient(f, x)
    epsilon = 1e-5;  % 小量，用于数值逼近
    n = length(x);
    grad = zeros(n, 1);
    for i = 1:n
        x_plus = x;
        x_plus(i) = x_plus(i) + epsilon;
        x_minus = x;
        x_minus(i) = x_minus(i) - epsilon;
        grad(i) = norm(f(x_plus) - f(x_minus)) / (2 * epsilon);
    end
end

