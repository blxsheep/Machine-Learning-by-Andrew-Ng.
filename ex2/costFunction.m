function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X*theta);

%size of h =  m x 1 
%size of y  = m x 1
%form equation sigma[-yi*log(h(xi))-(1-yi)*log(1-h(xi))]
% it's can replace with  sigma1[-yi*log(h(xi))] - sigma2[(1-yi)*log(1-h(xi))]
%so that you can implement each sigma as matrix multiplication

sigma1 = -y'*log(h);
sigma2 = (1-y)'*log(1-h);

equation = sigma1- sigma2 ;
J = equation/m;

grad  = ((h-y)'*X)/m;












% =============================================================

end
