function prob=logpdf(x,mu,sigma,d)
prob=-1/2*sum(bsxfun(@minus,x, mu).^2,2)/sigma-d/2*log(2*pi*sigma);
