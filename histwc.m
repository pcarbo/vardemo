function [h, int] = histwc (x, w, n)
  d   = (max(x) - min(x))/n;
  int = linspace(min(x), max(x), n) - d/2;
  h   = zeros(n,1);
  for i = 1:length(x)
    I = find(int < x(i),1,'last');
    if ~isempty(I)
      h(I) = h(I) + w(i);
    end
  end
end