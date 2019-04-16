clear;
clc;
close all;

% Weight 
% initialize with zero or 0.1
%init_w = 0;
init_w = 0.1;
W_1 = init_w*ones(2, 2);
W_2 = init_w*ones(1, 2);

% Bias
B_1 = ones(2, 1);
B_2 = ones(1, 1);

% Input
X = [-1 -1; -1 1; 1 -1; 1 1]';

% GT
D = [-1 1 1 -1];

J_array = [2];
% Training
epoch = 1;
for i = 1:epoch
    if mod(i, 100) == 0
        disp(['epoch: ' num2str(i)]);
    end
    [J, W_1_trained ,W_2_trained] = train_network(X, D, W_1, W_2, B_1, B_2, 0.001);
    J_array = [J_array J'];
    W_1 = W_1_trained;
    W_2 = W_2_trained;
end
plot(J_array);

% Test
[Z, J] = run_network(X, D, W_1, W_2, B_1, B_2);
