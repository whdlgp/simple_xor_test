function y = d_sigmoid(x)
% Differential of sigmoid
    y = (1 - sigmoid(x)).*sigmoid(x);
end

