# simple_xor_test
test Matlab source for xor problem with deeplearning, 

## Sigmoid and Cost function
### Sigmoid and differential

```matlab
function y = sigmoid(x)
% Sigmoid function
    y = 1 ./ (1 + exp(-x));
end
```
```matlab
function y = d_sigmoid(x)
% Differential of sigmoid
    y = (1 - sigmoid(x)).*sigmoid(x);
end
```

### Cost function and differential

```matlab
function J = cost_function(Z, D)
% cost function 
    J =0.5*pow2(Z - D);
end
```
```matlab
function J_diff = d_cost_function(Z, D)
% Differential of cost function
    J_diff = (Z - D);
end
```
## Network Train and Test
### Network for 1 step
```matlab
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
```
### Training 1 epoch
After launch 1 step of network, update weight with Back propagation,  
print out cost and updated weight and 
```matlab
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
```

### Test 1 epoch
Launch once for Test
```matlab
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
```

# Test result
## One epoch with initial weight 0.1, learning rate 0.1
```
*
*
step: 1
J = 2.2007
W_11_traind: 0.10457
W_12_traind: 0.10457
W_21_traind: 0.10457
W_21_traind: 0.10457
W_13_traind: -0.047516
W_23_traind: -0.047516
*
step: 2
J = 0.55333
W_11_traind: 0.10029
W_12_traind: 0.10029
W_21_traind: 0.099713
W_21_traind: 0.099713
W_13_traind: 0.089311
W_23_traind: 0.089311
*
step: 3
J = 0.55333
W_11_traind: 0.099713
W_12_traind: 0.099713
W_21_traind: 0.10029
W_21_traind: 0.10029
W_13_traind: 0.089311
W_23_traind: 0.089311
*
step: 4
J = 2.2248
W_11_traind: 0.096169
W_12_traind: 0.096169
W_21_traind: 0.096169
W_21_traind: 0.096169
W_13_traind: -0.065518
W_23_traind: -0.065518
*
test result
W_11: 0.096169
W_12: 0.096169
W_21: 0.096169
W_21: 0.096169
W_13: -0.065518
W_23: -0.065518
step 1, J = 1.8782
step 2, J = 0.46788
step 3, J = 0.46788
step 4, J = 1.8654
```

## One epoch with initial weight 0, learning rate 0.1
```
*
step: 1
J = 2
W_11_traind: 0
W_12_traind: 0
W_21_traind: 0
W_21_traind: 0
W_13_traind: -0.14621
W_23_traind: -0.14621
*
step: 2
J = 0.5
W_11_traind: 0
W_12_traind: 0
W_21_traind: 0
W_21_traind: 0
W_13_traind: 0
W_23_traind: 0
*
step: 3
J = 0.5
W_11_traind: 0
W_12_traind: 0
W_21_traind: 0
W_21_traind: 0
W_13_traind: 0
W_23_traind: 0
*
step: 4
J = 2
W_11_traind: 0
W_12_traind: 0
W_21_traind: 0
W_21_traind: 0
W_13_traind: -0.14621
W_23_traind: -0.14621
*
test result
W_11: 0
W_12: 0
W_21: 0
W_21: 0
W_13: -0.14621
W_23: -0.14621
step 1, J = 1.7246
step 2, J = 0.43114
step 3, J = 0.43114
step 4, J = 1.7246
```

## About result

When initial weight 0, We can see only W_13 and W_23 updated and W_11, W_12, W_21, W_21 are not updated.  
Because of Backpropagation, If gradient of Activation function or Weights are zero, weights with backpropagation will not be updated
