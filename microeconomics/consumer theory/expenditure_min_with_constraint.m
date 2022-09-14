%% Setup
clear; clc;
% Parameters
syms p1 p2 u0 y

% Choice variables
syms x1 x2 lambda

% Set real parameter values
u0_real = 5.5;
p1_real = 2;
p2_real = 1;

% Utility function
u = @(x1,x2) sqrt(x1) + x2;

% Lagrangian
lagrangian = p1*x1 + p2*x2 + lambda*(u0 - u(x1,x2))

%% First order conditions
d_lagrangian_x1 = diff(lagrangian, x1);
d_lagrangian_x2 = diff(lagrangian, x2);
d_lagrangian_lambda = diff(lagrangian, lambda);

GRADIENT = [d_lagrangian_x1; d_lagrangian_x2; d_lagrangian_lambda];

%% Identify optimums
% Substitute selected values for parameters
% eg, p1=1,p2=2,u0=5.5
GRADIENT = subs(GRADIENT, {p1,p2,u0}, {p1_real,p2_real,u0_real});

% Solve the Hicksian demands
hicksians = solve(GRADIENT(1)==0, GRADIENT(2)==0, GRADIENT(3)==0, x1, x2, lambda, 'Real', true);

h_x1 = double(hicksians.x1);
h_x2 = double(hicksians.x2);
h_lambda = double(hicksians.lambda);

% Display results
disp(table(h_x1, h_x2, h_lambda))

%% What is the gradient at the optimum?
gradientAtOptimum = double(subs(GRADIENT, {x1,x2,lambda}, {h_x1,h_x2,h_lambda}))

%% What is the Hessian at the optimum?
HESSIAN = [diff(GRADIENT(1),x1), diff(GRADIENT(2),x1);...
           diff(GRADIENT(1),x2), diff(GRADIENT(2),x2)]

HESSIANAtOptimum = double(subs(HESSIAN, {x1,x2,lambda}, {h_x1,h_x2,h_lambda}))

%% Definiteness of Hessian?
% Initialize a vector
xAxSave = zeros(1,100);

% Try for 100 random xs to see what xAx equals
for i = 1:100
    x = rand(1,2);
    xAxSave(i) = x*HESSIANAtOptimum*x';
end

% Check the first 5
xAxSave(1:5)

% Create a new logical vector if xAxSave is positive
is_positive = (xAxSave > 0);
sum(is_positive)

%% Calculate maximum utility
expenditure = p1_real*h_x1 + p2_real*h_x2