function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);
WT = W * ZCAWhite;
bWTm = b - WT * meanPatch;
t = patchDim^2;
convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
feature = cell(imageChannels, 1);
for featureNum = 1:numFeatures
    for channel = 1:imageChannels
        feature{channel} = reshape(WT(featureNum, t*(channel-1)+1:t*channel), patchDim, patchDim);
        feature{channel} = rot90(feature{channel}, 2);
    end
    for imageNum = 1:numImages
        convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
        for channel = 1:imageChannels
            im = squeeze(images(:, :, channel, imageNum));
            convolvedImage = convolvedImage + conv2(im, feature{channel}, 'valid');
        end
        convolvedFeatures(featureNum, imageNum, :, :) = sigmoid(convolvedImage + bWTm(featureNum));
    end
end
% not memory efficient!!!!!!
% convolvedFeatures = bsxfun(@plus, convolvedFeatures, b - WTm);
% convolvedFeatures = sigmoid(convolvedFeatures);
end
