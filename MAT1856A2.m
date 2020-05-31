
% MATH FIN Q1C DIAG CODE

P = [0.8,0.1,0.1,0;0.1,0.5,0.2,0.2;0.1,0.3,0.3,0.3;0,0,0,1]
[V,D]=eig(P);
A=V*D*inv(V);
%A = A + A.'; % make sure A is symmetric
% disp(A)
% [V,~]=eig(A);
% V.'*V

A1 = V*D^(1/12)*inv(V)

% m = [1,2;4,2]
% [z,q]=eig(m)
% 
% A2 = z*q^(1.5)*inv(z)
% 
% m^(1.5)


B=V*D^(100)*inv(V);
B1 = V*D^(101)*inv(V);

BN = V*D^(1000)*inv(V);



P^3
V*D^(3)*inv(V)

P^100

0.8^3

F = [0,0,0,0;0,0,0,0;0,0,0,0;0,0,0,1]
V*F*inv(V)

g = V*F

format long

