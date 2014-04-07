function numgrad = computeNumericalGradient(J, theta)
% numgrad = COMPUTENUMERICALGRADIENT(J, theta)
%
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will
%    return the function value at theta.

assert(isvector(theta), 'theta must be a vector.');
epsilon = 1e-4;
numgrad = zeros(size(theta));
perturb = zeros(size(theta));
for i = 1:numel(theta)
    perturb(i) = epsilon;
    numgrad(i) = (J(theta + perturb) - J(theta - perturb)) / (2*epsilon);
    perturb(i) = 0;
end
end