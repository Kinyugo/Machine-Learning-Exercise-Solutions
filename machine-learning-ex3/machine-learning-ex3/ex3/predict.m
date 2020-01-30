function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(size(X, 1), 1), X];

hidden_layer_size = size(Theta1, 1);
hidden_layer_output = zeros(size(X, 1), hidden_layer_size);

for a = 1:hidden_layer_size
  theta = Theta1(a, :);
  col_pred = sigmoid(X * theta(:));
  hidden_layer_output(:, a) = col_pred;
end

output_layer_neurons = size(Theta2, 1);
predictions = zeros(size(hidden_layer_output, 1), output_layer_neurons);

hidden_layer_output = [ones(size(hidden_layer_output, 1), 1), hidden_layer_output];



for c = 1:output_layer_neurons
  theta = Theta2(c, :);
  g = sigmoid(hidden_layer_output * theta(:));
  predictions(:, c) = g;
end
% =========================================================================
[max_values, max_indices] = max(predictions, [], 2);
p = max_indices;

end
