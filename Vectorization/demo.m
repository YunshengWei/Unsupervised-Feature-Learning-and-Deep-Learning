images = loadMNISTImages('../MNIST/train-images-idx3-ubyte');
labels = loadMNISTLabels('../MNIST/train-labels-idx1-ubyte');
 
% We are using display_network from the autoencoder code
display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

visibleSize = 28*28;
hiddenSize = 196;
sparsityParam = 0.1;
lambda = 3e-3;
beta = 3;
patches = images(:, 1:10000);

sparseAutoencoderModel = trainSparseAutoencoder(visibleSize, hiddenSize, lambda, ...
                                  sparsityParam, beta, patches);
                              
display_network(sparseAutoencoderModel.W1'); 
print -djpeg weights.jpg   % save the visualization to a file