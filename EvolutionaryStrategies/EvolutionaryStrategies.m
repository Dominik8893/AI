%% Assigning our Data to x and y
Round = 0;
Taul = 0.316; % n-5
Tau2 = 0.472;
plot(x, y);
hold on
z = zeros(1, 101);
NumberOfChildren = 500;
NumberOfGenerations = 2500;
ChildrenPerParent = 1;
ErrorMargin = 0.25;
%%%
Matrix = zeros(NumberOfChildren, 4);
% Create Matrix for our Results
% Defining a Function Handle to quickly calculate the Value f = @(a,b,c,x) a*(x.^2-b*cos(x*c*pi));
EndResults = zeros(10, 6);
while (Round < 10) %%%how many times we run the code to obtain an average Round Round+1;
    tic;
    % Creating Random ABC value at start of Each round
    for n = 1:NumberOfChildren
        for i = 2:4
            Matrix(n, i) = rand(1, 1);
            Matrix(n, i) = (Matrix(n, i) * 20) - 10;
        end
    end
    MSE1 = 9999;
    LastGenError = 10000;
    % disp(Matrix);
    %%
    % column 2 a
    % column 3 b
    % column 4 c
    % column 1 MSE
    MutationMatrix = zeros(NumberOfChildren, 3);
    for n = 1:NumberOfChildren
        for i = 1:3
            MutationMatrix(n, i) = rand(1, 1) * 10;
        end
    end
    for o = 1:NumberOfGenerations % start of Generation
        % Creating Mutation Values for Children
        % 1a 2b 3c
        for n = 1:NumberOfChildren
            Sigma1 = randn() * Taul;
            for i = 1:3
                Matrix(n, i + 1) = Matrix(n, i + 1) + randn() * MutationMatrix(n, i);
                Sigma2 = randn() * Tau2;
                MutationMatrix(n, i) = MutationMatrix(n, i) * exp(Sigma1) * exp(Sigma2);
            end
        end
        % disp(Sigma)
        MSE = zeros(1, NumberOfChildren);
        for n = 1:NumberOfChildren
            % calculate Value for eachPoint
            Matrix(n, 1) = mean((y - f(Matrix(n, 2), Matrix(n, 3), Matrix(n, 4), x)).^2);
        end
        % sort by Best MSE
        MSE1 = sortrows(Matrix);
        % Take Best Parent from Generation and Create Mutated Children CH = 1;
        for n = 1:NumberOfChildren
            for i = 2:4
                if (CH == ChildrenPerParent + 1)
                    CH = 1;
                end
                Matrix(n, i) = MSE1(CH, i);
                CH = CH + 1;
            end
        end
        % plot(x, z);
        % legend({'0', '1', '2', '3'}, 'Location', 'southwest') disp(o)
        disp(MSE1(1, :))
        %%%%
        %%%%
        % end of Program if stopping condition is met
        if (abs(MSE1(1, 1) - LastGenError) < ErrorMargin)
            break;
        else
            LastGenError = MSE(1, 1);
        end
    end
    for i = 1:101
        % calculate Value for eachPoint
        z(i) = MSE1(1, 2) * (x(i)^2 - MSE1(1, 3) * cos(x(i) * MSE1(1, 4) * pi));
    end
    plot(x, z);
    EndResults(Round + 1, 1) = MSE1(1, 1);
    EndResults(Round + 1, 2) = MSE1(1, 2);
    EndResults(Round + 1, 3) = MSE1(1, 3);
    EndResults(Round + 1, 4) = MSE1(1, 4);
    EndResults(Round + 1, 5) = 0;
    EndResults(Round + 1, 6) = toc;
end
disp("Results:")
disp("MSE")
disp(EndResults);
disp("Average MSE:")
disp(mean(EndResults(:, 1)))
disp("Average Generation:")
disp(mean(EndResults(:, 5)))
disp("Average Time:")
disp(mean(EndResults(:, 6)))
hold off
