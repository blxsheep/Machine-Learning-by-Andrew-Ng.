function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 =  [ones(size(X),1),X];
z2 = a1*Theta1' ; 
a2 = [ones(size(z2),1),sigmoid(z2)];
z3  = a2 *Theta2';
a3 =  sigmoid(z3);
% change y = [1,2,3,4,5] to Matrix if y(i) = 2 so y = [0,1,0,..0] for each rows
new_y = zeros(m,num_labels) ;
new_y(sub2ind(size(new_y), (1:length(y))', y)) = 1;

J = sum(sum(-new_y .* log(a3) - (1 - new_y) .* log(1 - a3), 2)) / m;

% Regualize
J += (sum(sum(Theta1(:,2:end).^2)) +  sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% for sample t th elelment
yb = new_y ; % Just rename varible
for t = 1:m;
    a1= X(t,:)' ;% select once a row.
    a1  = [1;a1];
    z2 = Theta1*a1;
    a2  = sigmoid(z2);
    a2  = [1;a2];
    z3= Theta2*a2;
    a3  = sigmoid(z3);
    
    %Calculate Error 
    yt = yb(t,:)'; % select once a row.
    delta_3 =  a3-yt ; 
    % Propagate Backward << 
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2]);
    delta_2 = delta_2(2:end); 
    % dt2 , dt1 is partial derivative result of Cost function(J(Theta)) 
    dt2 = delta_3 * a2'; 
    dt1 = delta_2 * a1';
 % Cumulative
    Theta2_grad += dt2;
    Theta1_grad +=Theta1_grad + dt1;
end 
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad ;

% Add regularization terms
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
 
    
     
    
    

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
