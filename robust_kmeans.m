function [L,iter] = robust_kmeans(x,k, m,delta, alpha)
% The k-means++ initialization.
L1=0;
[center,~]= kmeans_with_mixture(x, k, delta, alpha);
C= x(center,:);
[~,L] = max(bsxfun(@minus,2*real(C*x.'),dot(C',C',1).'));
% The k-means algorithm.
iter =0;
while any(L ~= L1)
    iter = iter +1;
    L1 = L;
    for i = 1:m, l = L==i; C(i,:) = sum(x(l,:),1)/sum(l); end
    [~,L] = max(bsxfun(@minus,2*real(C*x.'),dot(C',C',1).'),[],1);
end