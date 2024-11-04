function [center,label]=scrlm(x,nsub,k,rho,F,seed)
rng(seed);
thr=0;
[N,p]=size(x);
n=randperm(N);
n=n(1:nsub);
loss=pdist2(x(n,:),x);
loss=(loss.^2)/(p*rho^2)-F;
loss=min(loss,0);
sloss=sum(loss,2);
[~,idx]=sort(sloss);
idx(sloss(idx)>=-F)=[];% do not consider negatives
counter=1;
sel=[];
while ~isempty(idx)&&(counter<=k)
    i=idx(1);
    sel(counter)=i;
    idx(loss(idx,n(i))<thr) = []; % remove nearby obs
    counter=counter+1;
end
center= x(n(sel),:);
d=pdist2(center,x);
%d=(d.^2)/(p*rho^2)-F;
[~,label]=min(d);
%label(k>0)=-1;