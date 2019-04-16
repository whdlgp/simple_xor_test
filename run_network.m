function [Z, J] = run_network(X, D, W_1, W_2, B_1, B_2)
% Run network

    disp(['W_11: ' num2str(W_1(1, 1))]);
    disp(['W_12: ' num2str(W_1(2, 1))]);
    disp(['W_21: ' num2str(W_1(1, 2))]);
    disp(['W_21: ' num2str(W_1(2, 2))]);
    disp(['W_13: ' num2str(W_2(1))]);
    disp(['W_23: ' num2str(W_2(2))]);

    input_num = length(X);
    Z = zeros(input_num, 1);
    J = zeros(input_num, 1);

    % run with inputs
    for i = 1:input_num
        [~, ~, Z_tmp, J_tmp] = run_network_1step(X(:, i), D(i), W_1, W_2, B_1, B_2);
        % store outputs and costs
        Z(i) = Z_tmp;
        J(i) = J_tmp;
        disp(['step ' num2str(i) ', J = ' num2str(J(i))]);
    end
    
end

