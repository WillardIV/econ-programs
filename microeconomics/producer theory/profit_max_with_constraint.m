% Parameters
syms p A alpha beta w r q

% Choice variables
syms K L m

% Lagrangian
lagrangian = (p*A*(K^alpha)*(L^beta) - w*L - r*K) - m*(A*(K^alpha)*(L^beta) - q)

%% FOC
d_lagrangian_K = diff(lagrangian, K);
d_lagrangian_L = diff(lagrangian, L);
d_lagrangian_m = diff(lagrangian, m);

GRADIENT = [d_lagrangian_K; d_lagrangian_L; d_lagrangian_m]

%%
GRADIENT = subs(GRADIENT, {A,p,w,r,q,alpha,beta},{1,1,1,1,2,0.3,0.7});
solu = solve(GRADIENT(1)==0, GRADIENT(2)==0, GRADIENT(3)==0, K, L, m, 'Real', true);

soluK = double(solu.K);
soluL = double(solu.L);
soluM = double(solu.m);
disp(table(soluK, soluL, soluM));

%% What is the gradient at the optimum?
gradientAtOptimum = double(subs(GRADIENT, {K,L,m}, {soluK,soluL,soluM}))

%% What is the Hessian at the optimum?
HESSIAN = [diff(GRADIENT(1),K), diff(GRADIENT(2),K);...
           diff(GRADIENT(1),L), diff(GRADIENT(2),L)];
HESSIANatOptimum = double(subs(HESSIAN, {K,L,m}, {soluK,soluL,soluM}))

%% Is the Hessian positive definite or negative definite?
xAxSave = zeros(1,100);
% Try 100 random xs to see what xAx equals
for i=1:100
    x = rand(1,2);
    xAxSave(i) = x*HESSIANatOptimum*x';
end

% Check the first 5
xAxSave(1:5)

% Create a new logical vector if xAxSave is negative
is_negative = (xAxSave < 0);
is_negative(1:5)
sum(is_negative)