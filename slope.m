function y = slope (x)
  y = (sigmoid(x) - 0.5)./x;
