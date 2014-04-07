function opttheta = fineTuning(theta, inputSize, hiddenSize, numClasses, ...
                    netconfig, lambda, data, labels, method, maxIter)
% opttheta = FINETUNIG(theta, hiddenSize, numClasses,
%            netconfig, lambda, data, labels, method, maxIter)
% theta: trained weights from the autoencoder
% inputSize :  the number of input units
% hiddenSize:  the number of hidden units *at the last autoencoder layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i)
%       is the i-th training example.
% labels: A vector containing labels, where labels(i) is the label for the
%         i-th training example
% method: used to compute descent direction ('lbfgs')
% maxIter: Maximum number of iterations allowed (400)

if ~exist('method', 'var') || isempty(method)
    method = 'lbfgs';
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = 400;
end    

addpath ../minFunc/
options.Method = method; % Here, we use L-BFGS to optimize our cost function.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc(@(p) stackedAECost(p, inputSize, hiddenSize, ...
                   numClasses, netconfig, lambda, data, labels), theta, options);

end