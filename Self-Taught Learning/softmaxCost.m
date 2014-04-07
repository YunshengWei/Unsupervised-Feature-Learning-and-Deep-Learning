function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
% [cost, grad] = SOFTMAXCOST(theta, numClasses, inputSize, lambda, data, labels)
%
% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds
%        to a single test set
% labels - an M x 1 matrix containing the labels corresponding for the
%          input data
%
% The input theta is a vector (because minFunc expects the parameters to be
% a vector).

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));

M = theta * data;
M = exp(bsxfun(@minus, M, max(M, [], 1)));
M = bsxfun(@rdivide, M, sum(M, 1));
cost = -mean(sum(groundTruth .* log(M), 1)) ...
    + lambda/2 * sum(sum(theta.^2));
thetagrad = 1/numCases * (M - groundTruth) * data' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);

end

