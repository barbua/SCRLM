function data = GMM_with_Outlier_fixed(N,p,m,sigma,w)
rng(0); 
count = mnrnd(N,w);
y=zeros(N,1);
x=zeros(N,p);
for i=1:m
    Sigma=sigma(i);
    SIGMA=Sigma^2*eye(p);
    pos=mvnrnd(zeros(p,1),eye(p));
    j=1+sum(count(1:i-1)):sum(count(1:i));
    x(j,:)=mvnrnd(pos,SIGMA,count(i));
    y(j,:)=i;
end
j=1+sum(count(1:m)):sum(count(1:m+1));
x(j,:)=mvnrnd(zeros(p,1),eye(p),count(m+1));
y(j,:)=-1;
data=[x,y];

