%Machine Learning Project
%Written by Ryan Norman and Sarah Garber

data = dlmread('creditcard.csv', ',', 1, 0);
x = data(:, 1:30);
y = data(:, 31);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(x);

% Add intercept term to x and X_test
x = [ones(m, 1) x];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, x, y);
%  Run fminunc to obtain the optimal theta
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, x, y)), initial_theta, options);

fprintf('theta: \n');
fprintf(' %f \n', theta);

% Compute accuracy on our training set
p = predict(theta, x);



