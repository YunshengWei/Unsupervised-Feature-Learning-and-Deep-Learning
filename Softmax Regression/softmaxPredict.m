function pred = softmaxPredict(softmaxModel, data)
% pred = SOFTMAXPREDICTION(softmaxModel, data)
% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% pred, where pred(i) is argmax_c P(y(c) | x(i)).

[Y, pred] = max(softmaxModel.optTheta * data, [], 1);

end

