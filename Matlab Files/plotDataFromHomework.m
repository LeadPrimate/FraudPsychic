function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 

% Creating graph
figure; hold on;

% Plotting on 2D plot
pos = find(y == 1);
neg = find(y == 0);

% adjusting the values of the axes
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

% ************************************************ %

end
