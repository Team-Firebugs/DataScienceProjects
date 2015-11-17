function g = sigmoidg(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = sigmoid(z).*(1-sigmoid(z));
end
