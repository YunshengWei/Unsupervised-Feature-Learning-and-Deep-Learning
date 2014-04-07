function y = sigmoidGradient(x)
% SIGMOIDGRADIENT(x) returns the gradient of the sigmoid function evaluated
% at x

    y = sigmoid(x) .* (1 - sigmoid(x));

end