function [activation] = feedForwardAutoencoder(sparseAEModel, hiddenSize, visibleSize, data)
% [activation] = FEEDFORWARDAUTOENCODER(sparseAEModel, hiddenSize, visibleSize, data)

% sparseAEModel: trained model from the sparseAutoencoder
% visibleSize: the number of input units
% hiddenSize: the number of hidden units
% data: Our matrix containing the training data as columns.  So, data(:,i)
%       is the i-th training example.
  
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

% W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

assert(sparseAEModel.hiddenSize == hiddenSize ...
       && sparseAEModel.visibleSize == visibleSize, ...
       'Dimensions of model and dimensions of data do not match.');
activation = sigmoid(bsxfun(@plus, sparseAEModel.W1 * data, sparseAEModel.b1));

end

