function sparseAEModel = trainSparseAutoencoder(visibleSize, hiddenSize, lambda, ...
                         sparsityParam, beta, data, method, maxIter)
% sparseAEModel = TRAINSPARSEAUTOENCODER(visibleSize, hiddenSize, lambda, sparsityParam, beta, data)
%
% visibleSize: the number of input units
% hiddenSize: the number of hidden units
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units
% beta: weight of sparsity penalty term
% data: data(:,i) is the i-th training example
% method: used to compute descent direction ('lbfgs')
% maxIter: Maximum number of iterations allowed (400)

if ~exist('method', 'var') || isempty(method)
    method = 'lbfgs';
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = 400;
end    

theta = initializeParameters(hiddenSize, visibleSize);
addpath ../minFunc/
options.Method = method; % Here, we use L-BFGS to optimize our cost function.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc(@(p) sparseAutoencoderCost(p, visibleSize, hiddenSize, ...
                           lambda, sparsityParam, beta, data), theta, options);
                       
sparseAEModel.W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
sparseAEModel.b1 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
sparseAEModel.W2 = reshape(opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
sparseAEModel.b2 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);
sparseAEModel.visibleSize = visibleSize;
sparseAEModel.hiddenSize = hiddenSize;

end