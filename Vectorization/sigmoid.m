function y = sigmoid(x)
% SIGMOID(x) is the sigmoid function of x
    
    y = 1 ./ (1 + exp(-x));

end