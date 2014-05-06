% Demonstrates (1) importance sampling, (2) Laplace's method and (3) a
% variational approximation for a simple logistic regression model with an
% intercept, and a single predictor. In logistic regression, we model Y (the
% binary outcome, or "coin toss") as a linear combination of the
% predictor(s) or explanatory variable(s) X:
%
%         p(yi = 1)
%     log --------  =  mu + xi*beta
%         p(yi = 0)      
%
% where yi is the ith observation of the binary outcome, xi is the ith
% observation of some quantity that might help predict the binary
% outcome, mu is the intercept, and beta is the coefficient
% corresponding to the variable X.
%
% Here I assume that the intercept (mu) is known, and doesn't have to be
% estimated, so we only need to estimate the regression coefficient beta.
%
%   Copyright (c) Peter Carbonetto, 2014
%   Dept. of Human Genetics
%   University of Chicago
% 
clear

% SCRIPT PARAMETERS
% -----------------
beta  = 2;    % True effect, or "log-odds ratio".
beta0 = -1;   % True intercept.
n     = 16;   % Number of observations.
ns    = 1e4;  % Number of Monte Carlo samples.
a     = 10;   % Draw Monte Carlo samples uniformly over [-a,a].
s0    = 10^2; % Prior variance of beta.

% Set the random number generator seed.
seed = 1;
rng(seed);

% GENERATE DATA SET
% -----------------
x = randn(n,1);
x = x - mean(x);
y = rand(n,1) < sigmoid(beta0 + x*beta);

% (1) IMPORTANCE SAMPLING
% -----------------------
% Draw Monte Carlo samples uniformly between -a and a. (This is actually a
% bit of a cheat because I'm not drawing the samples randomly, but should
% give a similar result.) 
samples = linspace(-a,a,ns);

% Compute the (unnormalized) importance weights.
w = zeros(ns,1);
for i = 1:ns
  p    = sigmoid(beta0 + x*samples(i));
  w(i) = exp(y'*log(p) + (1-y)'*log(1-p) ...
             - samples(i)^2/(2*s0)) / sqrt(2*pi*s0);
end

% Compute the importance sampling estimate of the marginal log-likelihood
% (i.e. the logarithm of the normalizing constant). As the number of samples
% (n) approaches infinity, sum(w)/n approaches Z/Z*, where Z is the
% normalizing constant of the target density, and Z* is the normalizing
% constant of the proposal density. Here Z* = 2*a, because we the proposal
% density is the uniform distribution.
I = log(2*a*mean(w));

% Normalize the importance weights.
w = w / sum(w);

% Plot a histogram of the importance weights.
set(gcf,'Color','white');
subplot(1,2,1);
[h int] = histwc(samples,w,50);
bar(int,h,1,'EdgeColor',rgb('darkorange'),'FaceColor',rgb('darkorange'));
set(gca,'FontSize',10,'FontName','fixed');
xlabel('beta');
ylabel('posterior probability');
title('Monte Carlo');
set(gca,'XLim',[-2 6],'XTick',-5:5);

% (2) LAPLACE'S METHOD
% --------------------
% Compute the Laplace approximation to the marginal log-likelihood, in
% which the Taylor series approximation is centered at the maximum a
% posteriori estimator.
[ans i] = max(w);
betamap = samples(i);
p       = sigmoid(beta0 + x*betamap);
W       = diag(sparse(p.*(1-p)));
s       = 1/(x'*W*x);
Ilap    = log(s/s0)/2 + y'*log(p) + (1-y)'*log(1-p);

% Plot the Laplace approximation.
hold on
r = exp(-(samples - betamap).^2/(2*s));
r = max(h)/max(r)*r;
plot(samples,r,'-','LineWidth',2,'Color',rgb('royalblue'));
hold off

% (3) VARIATIONAL APPROXIMATION
% -----------------------------
% Compute the variational approximation.
theta = ones(n,1);
i     = 0;
while true
  i = i + 1;
  theta0 = theta;

  % E-step.
  u    = slope(theta);
  U    = diag(sparse(u));
  yhat = y - 0.5 - beta0*u;
  s    = 1./(x'*U*x);
  mu   = s*x'*yhat;

  % Compute the variational lower bound.
  Ivar(i) = log(s/s0)/2 + theta'*(u.*theta - 1)/2 + beta0^2*sum(u)/2 ...
		  + sum(log(sigmoid(theta))) + beta0*sum(yhat) ...
		  + mu.^2/(2*s);

  % M-step.
  theta = sqrt((beta0 + mu*x).^2 + s*x.^2);

  % Check convergence.
  if max(abs(theta0 - theta)) < 0.001;
    break
  end
end

% Plot the variational approximation.
hold on
r = exp(-(samples - mu).^2/(2*s));
r = max(h)/max(r)*r;
plot(samples,r,'-','LineWidth',2,'Color',rgb('firebrick'));
hold off

% Plot the evolution of the variational lower bound.
subplot(1,2,2);
plot([1 length(Ivar)],[I I],'--','LineWidth',2,'Color',rgb('navy'));
hold on
plot([1 length(Ivar)],[Ilap Ilap],'LineWidth',2,...
     'Color',rgb('darkorange'));
plot(1:i,Ivar,'-','LineWidth',2,'Color',rgb('firebrick'));
hold off
set(gca,'XLim',[1 length(Ivar)]);
set(gca,'FontSize',10,'FontName','fixed');
xlabel('iteration');
ylabel('marginal log-likelihood');
title('Estimate of marginal likelihood');