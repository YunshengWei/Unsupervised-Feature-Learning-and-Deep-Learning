function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
% [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)                                         
% stackedAEPredict: Takes a trained theta and a test data set,
%                   and returns the predicted labels for each example.
%                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i)
%       is the i-th training example.
% pred: where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

activation = data;
sparseAEModel = struct;
for d = 1:numel(stack)
    sparseAEModel.W1 = stack{d}.w;
    sparseAEModel.b1 = stack{d}.b;
    sz = size(stack{d}.w);
    sparseAEModel.hiddenSize = sz(1);
    sparseAEModel.visibleSize = sz(2);
    activation = feedForwardAutoencoder(sparseAEModel, sz(1), sz(2), activation);
end

softmaxModel = struct;
softmaxModel.optTheta = softmaxTheta;
pred = softmaxPredict(softmaxModel, activation);

end

