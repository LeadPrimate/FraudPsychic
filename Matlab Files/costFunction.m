function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes a scalar value
%   w.r.t. to the parameters.

% Initializing values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Implienting Cost Function %
 h = sigmoid(X*theta);
 J = ((-y)'*log(h)-(1-y)'*log(1-h))/m;
 grad = (X'*(h-y))/m;

% *************************** %
end