%Machine Learning Project
%Written by Ryan Norman and Sarah Garber

data = dlmread('creditcard.csv', ',', 1, 0);
x = data(:, 1:30);
y = data(:, 31);

%  Setup the data matrix appropriately, and add ones for the intercept term
[dataHeight, dataWidth] = size(x);

% Add intercept term to x and X_test
x = [ones(dataHeight, 1) x];

% Initialize fitting parameters
initial_theta = zeros(dataWidth + 1, 1);

halfData = dataHeight/2;
training = x(0:halfData, :);
trainingValid = y(0:halfData, :);

[cost, grad] = costFunction(initial_theta, training, trainingValid);
%  Run fminunc to obtain the optimal theta
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, x, y)), initial_theta, options);

fprintf('theta: \n');
fprintf(' %f \n', theta);


testing = x(halfData:dataHeight, :);
testingValid = y(halfData:dataHeight, :);

p = predict(theta, testing);

hitMiss = zeros(round(halfData), 1);
for n = 1 : size(p)
    if p(n) == testingValid(n)
        hitMiss(n) = 1;
    end
    if p(n) ~= testingValid(n)
        hitMiss(n) = 0;
    end
end

accuracy = sum(hitMiss)/ halfData;