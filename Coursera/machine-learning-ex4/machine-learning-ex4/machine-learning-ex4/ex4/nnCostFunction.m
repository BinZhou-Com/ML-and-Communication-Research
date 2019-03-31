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
X = [ones(m,1) X];      
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



a_1 = X';
z_2 = Theta1*a_1;
a_2 = sigmoid(z_2);
z_3 = Theta2*[ones(size(a_2,2), 1)'; a_2];
h = sigmoid(z_3);

% sum over number of labels
for k = 1:num_labels
    yk = (y == k);
    J = J + 1/m * sum(-log(h(k,:)).*yk' - log(1-h(k,:)).*(1-yk')); 
    % Isolate each training example and use .* mult
end

% Regularization
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + ...
    sum(sum(Theta2(:,2:end).^2)));

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
%               first time"

% Initialization of Delta
Delta1 = zeros(size(Theta1_grad));
Delta2 = zeros(size(Theta2_grad));

for t = 1:m % each training example
    % Step 1: forward propagation
    a_1 = X(t,:)';
    z_2 = Theta1*a_1;
    a_2 = sigmoid(z_2);
    a_2 = [ones(size(a_2,2), 1)'; a_2];
    z_3 = Theta2*a_2;
    h = sigmoid(z_3);
    
    % Step 2: set the error for each output unit
    delta3 = zeros(num_labels, 1);
    
    for k = 1:num_labels
        yk = 1.0*(y(t)==k);
        delta3(k) = h(k)-1.0*yk;
    end
    
     % Step 3: set the error for the hidden layer
     delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z_2);
     
     % Step 4: accumulate the gradient 
     Delta1 = Delta1 + delta2*a_1';
     Delta2 = Delta2 + delta3*a_2';
     
end

% Step 5: compute the derivative
Theta1_grad =  1/m * Delta1;
Theta2_grad =  1/m * Delta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Step 6: Regularizing the gradient
Theta1_grad(:,1) = Theta1_grad(:,1); % bias (first column)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);

Theta2_grad(:,1) = Theta2_grad(:,1); % bias (first column)
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
