function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Forward propaagation
m = size(data, 2);
z2 = bsxfun(@plus, W1 * data, b1);
a2 = sigmoid(z2);
averageRho = mean(a2, 2);
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = z3;
% mean(sum(a.^2)) is faster than norm(a,'fro')^2
cost = mean(sum((a3 - data).^2, 1)) / 2 ...
     + lambda/2 * (sum(sum(W1.^2)) + sum(sum(W2.^2))) ...
     + beta * sum(KL(sparsityParam, averageRho));
 
 % Backpropagation
delta3 = -(data - a3);
sparsityDelta = -sparsityParam ./ averageRho + (1 - sparsityParam) ./ (1 - averageRho);
delta2 = bsxfun(@plus, W2' * delta3, beta * sparsityDelta) .* sigmoidGradient(z2);
W2grad = delta3 * a2' / m + lambda * W2;
W1grad = delta2 * data' / m + lambda * W1;
b2grad = mean(delta3, 2);
b1grad = mean(delta2, 2);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
