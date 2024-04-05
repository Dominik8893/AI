function [Uh, Vh, J1, it] = HCM(c, epsilon)
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
%Iterative algorithm
while criterion > epsilon
it = it + 1;
for i = 1:c
for j = 1:dim
Nominator = 0;
Denominator = 0;
for k = 1:N
Nominator = Nominator + Uh(i, k) * X(j, k);
Denominator = Denominator + Uh(i, k);
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
for i = 1:c
for j = 1:N
J1 = J1 + Uh(i, j) * d(i, j);
end
end
Uh_old = Uh;
for i = 1:N
minimum = min(d(:, i));
for j = 1:c
if d(j, i) == minimum
Uh(j, i) = 1;
else
Uh(j, i) = 0;
end
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
%Create the plot
figure
for i = 1:c
temp_x = 1;
for j = 1:N
if Uh(i, j) == 1
x(temp_x) = X(1, j);
y(temp_x) = X(2, j);
temp_x = temp_x + 1;
end
end
plot(x(:), y(:), '.', 'MarkerSize', 8)
hold on;
end
x2 = Vh(1, :);
y2 = Vh(2, :);
plot(x2, y2, '.', 'MarkerSize', 8)
end
