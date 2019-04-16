function [J, W_1_trained ,W_2_trained] = train_network(X, D, W_1, W_2, B_1, B_2, learning_rate)
% Train weights 

    input_num = length(X);
    J = zeros(input_num, 1);
    for i = 1:input_num
        [L, Y, Z, J_tmp] = run_network_1step(X(:, i), D(i), W_1, W_2, B_1, B_2);
        % store outputs and costs

        W_2_trained = W_2 - learning_rate...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                           *d_cost_function(Z, D(i))...Differential of cost function
                           *Y'; %Differential of Z

        W_1_trained = W_1 - learning_rate...
                           *d_cost_function(Z, D(i))...Differential of cost function
                           *W_2'...Differential of Z
                           .*d_sigmoid(L)...Differential of sigmoid
                           *X(:, i)'; %Differential of L
        J(i) = J_tmp;
        
        disp('*');
        disp(['step: ' num2str(i)]);
        
        disp(['J = ' num2str(J(i))]);

        disp(['W_11_traind: ' num2str(W_1_trained(1, 1))]);
        disp(['W_12_traind: ' num2str(W_1_trained(2, 1))]);
        disp(['W_21_traind: ' num2str(W_1_trained(1, 2))]);
        disp(['W_21_traind: ' num2str(W_1_trained(2, 2))]);
        disp(['W_13_traind: ' num2str(W_2_trained(1))]);
        disp(['W_23_traind: ' num2str(W_2_trained(2))]);
    end
end

