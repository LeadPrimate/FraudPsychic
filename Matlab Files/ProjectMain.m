%Machine Learning Project
%Written by Ryan Norman and Sarah Garber

data = dlmread('creditcard.csv', ',', 1, 0);
x = data(:, 1:30);
y = data(:, 31);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(x);

% Add intercept term to x and X_test
x = [ones(m, 1) x];

% K-fold Cross Validation
folds = 5; % kPartion should be > 1
if folds <= 1
   error("Warning! folds has been set to a value below 2."); 
end

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);


indices = crossvalind('Kfold',data,folds);
cp = classperf(data);
for i = 1:folds
    test = (indices == i); train = ~test;
    class = classify(meas(test,:),meas(train,:),data(train,:));
    classperf(cp,class,test)
end


%for i = 1:kPartition
%    [cost, grad] = costFunction(...
%        initial_theta, x(partitionSize * i , :), y(partitionSize * i, :));
%    %  Run fminunc to obtain the optimal theta
%    options = optimset('GradObj', 'on', 'MaxIter', 400);
%    [theta, cost] = ...
%        fminunc(@(t)(costFunction(t, x, y)), initial_theta, options);
%end



