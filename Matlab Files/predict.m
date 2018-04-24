function [p, hxReturnVector] = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% return these variable
p = zeros(m, 1);

hxReturnVector = zeros(m, 1);

% Making predictions
boundry = 0.5;

for n = 1 : m
    
    hx = sigmoid(X(n,:) * theta);
    
    hxReturnVector(n) = hx;
    
    if hx >= boundry
        p(n) = 1;
    else
        p(n) = 0;
    end
end

% ************************************************ %

end
