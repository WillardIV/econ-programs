%% Setup
clear; clc;
% Parameters
syms p1 p2 u0 y

% Choice variables
syms x1 x2 lambda

% Utility function
u = @(x1,x2) sqrt(x1) + x2;

% Lagrangian
lagrangian = u(x1,x2) + lambda*(y - p1*x1 - p2*x2)

%% First order conditions
d_lagrangian_x1 = diff(lagrangian, x1);
d_lagrangian_x2 = diff(lagrangian, x2);
d_lagrangian_lambda = diff(lagrangian, lambda);

GRADIENT = [d_lagrangian_x1; d_lagrangian_x2; d_lagrangian_lambda];

%% Identify optimums
% Substitute selected values for parameters
% eg, p1=1,p2=2,y=10
GRADIENT = subs(GRADIENT, {p1,p2,y}, {1,2,10});

% Solve the Marshallian demands
marshallians = solve(GRADIENT(1)==0, GRADIENT(2)==0, GRADIENT(3)==0, x1, x2, lambda, 'Real', true);

m_x1 = double(marshallians.x1);
m_x2 = double(marshallians.x2);
m_lambda = double(marshallians.lambda);

% Display results
disp(table(m_x1, m_x2, m_lambda))

%% What is the gradient at the optimum?
gradientAtOptimum = double(subs(GRADIENT, {x1,x2,lambda}, {m_x1,m_x2,m_lambda}))

%% What is the Hessian at the optimum?
HESSIAN = [diff(GRADIENT(1),x1), diff(GRADIENT(2),x1);...
           diff(GRADIENT(1),x2), diff(GRADIENT(2),x2)];

HESSIANAtOptimum = double(subs(HESSIAN, {x1,x2,lambda}, {m_x1,m_x2,m_lambda}))

%% Definiteness of Hessian?
% Initialize a vector
xAxSave = zeros(1,100);

% Try 100 random xs to see what xAx equals
for i=1:100
    x = rand(1,2);
    xAxSave(i) = x*HESSIANAtOptimum*x';
end

% Check the first 5
xAxSave(1:5) % negative

% Create a new logical vector if xAxSave is negative
is_negative = (xAxSave < 0);
sum(is_negative)

%% Calculate maximum utility
value = u(m_x1,m_x2)