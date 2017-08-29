function [error_train, error_val] = ...
    learningCurveRandomized(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)). The i examples are
%       chosen randomly 50 times and the error values averaged.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ---------------------- Sample Solution ----------------------
random_runs = 50;
for c = 1:m
    for r = 1:random_runs
        rand_rows = randperm(m);
        Xtrain = X(rand_rows(1:c), :);
        ytrain = y(rand_rows(1:c), :);
        theta = trainLinearReg([ones(c, 1) Xtrain], ytrain, lambda);
        error_train(c) = error_train(c) ...
            + linearRegCostFunction([ones(c, 1) Xtrain], ytrain, theta, 0);
        error_val(c) = error_val(c) ...
            + linearRegCostFunction([ones(size(Xval,1), 1) Xval], yval, theta, 0);
    end
    error_train(c) = error_train(c) / random_runs;
    error_val(c) = error_val(c) / random_runs;
end




% -------------------------------------------------------------

% =========================================================================

end
