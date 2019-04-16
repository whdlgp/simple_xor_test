function [L, Y, Z, J] = run_network_1step(X, D, W_1, W_2, B_1, B_2)
% Run network with one set of input
    % First layer
    L = W_1*X + B_1;
    Y = sigmoid(L);

    % Second layer
    Z = W_2*Y + B_2;
    
    % Cost function
    J = cost_function(Z, D);
end

