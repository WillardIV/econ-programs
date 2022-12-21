clear; clc;

a1 = 2;
a2 = 3;
b1 = 0.5;
b2 = 1-b1;
x1 = linspace(a1,a1+100,100);
x2 = linspace(a2,a2+100,100);
[X1,X2] = meshgrid(x1,x2);

u = @(x1,x2) (x1-a1).^b1 .* (x2-a2).^b2;

contour(X1,X2,u(X1,X2));
xlabel('X1'); ylabel('X2'); zlabel('Direct Utility');