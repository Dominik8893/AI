function [Uh, Vh, J1, it] = FCM(c, m, epsilon)
%Load data
X = [2.5 3.0 3.0 3.5 5.5 6.0 6.0 6.5;
3.5 3.0 4.0 3.5 5.5 6.0 5.0 5.5];
[dim, N] = size(X);
%Randomize the partition matrix
Uh = zeros(c, N);
temp = 1;
for i = 1:N
Uh(temp, i) = 1;
temp = temp + 1;
if temp > c
temp = 1;
end
end
Uh = Uh(:, randperm(size(Uh, 2)));
Vh = zeros(dim, c);
criterion = 1000;
it = 0
Iterative algorithm
while criterion > epsilon
it = it + 1;
for i = 1:c
for j = 1:dim
Nominator = 0;
Denominator = 0;
for k = 1:N
Nominator = Nominator + Uh(i, k) ^ m * X(j, k);
Denominator = Denominator + Uh(i, k) ^ m;
end
if Denominator ~= 0
Vh(j, i) = Nominator / Denominator;
end
end
end
d = zeros(c, N);
for i = 1:c
for j = 1:N
for k = 1:dim
d(i, j) = d(i, j) + (X(k, j) - Vh(k, i)) ^ 2;
end
end
end
J1 = 0;
Uh_old = Uh;
for i = 1:c
for j = 1:N
J1 = J1 + Uh(i, j) ^ m * d(i, j);
Nominator = 0;
for k = 1:c
Nominator = Nominator + d(k, j) ^ (1 / (1 - m));
end
Uh(i, j) = d(i, j) ^ (1 / (1 - m)) / Nominator;
end
end
temp = Uh - Uh_old;
criterion = 0;
for i = 1:c
for j = 1:N
criterion = criterion + (temp(i, j)) ^ 2;
end
end
criterion = sqrt(criterion);
end
