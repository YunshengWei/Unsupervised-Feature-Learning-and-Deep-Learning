function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, ...
                                       inputData, labels, method, maxIter)
% softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the corresponding
%         inputs. labels(c) is the class label for the cth input
% method: used to compute descent direction ('lbfgs')
% maxIter: Maximum number of iterations allowed (400)

if ~exist('method', 'var') || isempty(method)
    method = 'lbfgs';
end
if ~exist('maxIter', 'var') || isempty(maxIter)
    maxIter = 400;
end    

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
addpath ../minFunc/
options.Method = method; % Here, we use L-BFGS to optimize our cost function.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';

[softmaxOptTheta, cost] = minFunc(@(p) softmaxCost(p, numClasses, inputSize, ...
                                  lambda, inputData, labels), theta, options);

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          
