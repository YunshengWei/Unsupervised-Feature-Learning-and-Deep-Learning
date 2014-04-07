function y = KL(rho, x)
% KL(rho, x) is the Kullback-Leibler divergence between a Bernoulli random
% variable with mean rho and a Bernoulli random variable with mean x
%
% KL-divergence is a standard function for measuring how different two
% distributions are.

    % assert(isscalar(rho), 'rho must be a scalar.');
    y = rho .* log(rho ./ x) + (1 - rho) .* log((1 - rho) ./ (1 - x));

end