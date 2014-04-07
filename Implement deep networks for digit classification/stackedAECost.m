function [cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                         numClasses, netconfig, lambda, data, labels)
% [cost, grad ] = stackedAECost(theta, hiddenSize, numClasses,
%                               netconfig, lambda, data, labels)
%
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
%                                         
% theta: trained weights from the autoencoder
% inputSize:   the number of input units
% hiddenSize:  the number of hidden units *at the last autoencoder layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i)
%       is the i-th training example.
% labels: A vector containing labels, where labels(i) is the label for the
%         i-th training example


% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

stackgrad = cell(size(stack));

numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));

% Forward propagation
depth = numel(stack) + 1;
z = cell(depth, 1);
a = cell(depth, 1);
a{1} = data;
for d = 2:depth
    z{d} = bsxfun(@plus, stack{d-1}.w * a{d-1}, stack{d-1}.b);
    a{d} = sigmoid(z{d});
end
M = softmaxTheta * a{depth};
M = exp(bsxfun(@minus, M, max(M, [], 1)));
M = bsxfun(@rdivide, M, sum(M, 1));
cost = -mean(sum(groundTruth .* log(M), 1)) + lambda/2 * sum(sum(softmaxTheta.^2));

% Backpropagation
delta = cell(depth,1);
softmaxThetaGrad = 1/numCases * (M - groundTruth) * a{depth}' + lambda * softmaxTheta;
delta{depth} = (softmaxTheta'*(M - groundTruth)) .* sigmoidGradient(z{depth});
for d = depth-1:2
    stackgrad{d}.w = delta{d+1} * a{d}' / numCases;
    stackgrad{d}.b = mean(delta{d+1}, 2);
    delta{d} = (stack{d}.w' * delta{d+1}) .* sigmoidGradient(z{d});
end
stackgrad{1}.w = delta{2} * a{1}' / numCases;
stackgrad{1}.b = mean(delta{2}, 2);

% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

