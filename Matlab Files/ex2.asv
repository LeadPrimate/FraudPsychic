%Machine Learning Project
%Written by Ryan Norman and Sarah Garber

data = dlmread('creditcard.csv', ',', 1, 0);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(data);

% Add intercept term to x and X_test
Time = [ones(m, 1) Time];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 31);


%  Run fminunc to obtain the optimal theta
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, Time, V1)), initial_theta, options);

fprintf('theta: \n');
fprintf(' %f \n', theta);


prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy on our training set
p = predict(theta, Time);



