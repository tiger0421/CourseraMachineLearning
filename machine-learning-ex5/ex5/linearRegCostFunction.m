function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;    %X: 12x2  theta: 2x1  h: 12x1
err = h - y;      %err: 12x1
thetaTemp = theta;
thetaTemp(1, :) = 0;

reg = lambda / (2*m) * (thetaTemp' * thetaTemp);  %reg: 1x1

J = 1 / (2*m) * sum(err.^2) + reg;


grad = 1/m * (err' * X)' + (lambda / m .* thetaTemp);  %grad: 2x1

% =========================================================================

grad = grad(:);

end