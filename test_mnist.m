clc;
clear;
addpath('..\..\Datasets\')
load("mnist.mat");
addpath('..\test_matlab\')
label = double(label);
sample = double(sample);
data=[sample,label];
p=512;
N=60000;
x=data(1:N,1:p);
y=data(1:N,p+1:end);
m=10;
% alpha is the coefficient of Uniform Sampling. 
% If alpha = 0, then k-means++ if alpha = 1, then uniform
k=4;
delta = 1/3;
it = 20;
Acc_robust_kmeans=zeros(it,1);
t_robust_kmeans=zeros(it,1);
t_kmeans = zeros(it,1);
Acc_kmeans = zeros(it,1);
t_scrlm_kmeans =  zeros(it,1);
Acc_scrlm_kmeans =  zeros(it,1);
t_scrlm =  zeros(it,1);
t_em=zeros(it,1);
t_cl=zeros(it,1);
t_sc=zeros(it,1);
t_tsne=zeros(it,1);
t_td=zeros(it,1);
Acc_scrlm=zeros(it,1);
Acc_cl=zeros(it,1);
Acc_em=zeros(it,1);
Acc_sc=zeros(it,1);
Acc_tsne=zeros(it,1);
Acc_td=zeros(it,1);
alpha = 0.5;
x=(x-mean(x))./std(x);
x=x./sqrt(p);
F=8/3;
n=100;
rho = 0.7/sqrt(p); 
for i=1:it
    rng(i);
    t1=tic;
    idx_kmeans=kmeans(x,m,'MaxIter',1000,'Display','final');
    t_kmeans(i)=toc(t1);
    Acc_kmeans(i)=accuracy(y,idx_kmeans);
    
    t2=tic;
    t_sne_method = tsne(x);
    idx_tsne = kmeans(t_sne_method,m,'MaxIter',1000,'Display','final');
    t_tsne(i)=toc(t2);
    Acc_tsne(i)=accuracy(y,idx_tsne);

    t3=tic;
    Z = linkage(x,'complete');
    idx_linkage=cluster(Z,m);
    t_cl(i)=toc(t3);
    Acc_cl(i)=accuracy(y,idx_linkage);

    t4=tic;
    gm = fitgmdist(x,m,'CovType','diagonal','RegularizationValue',0.001);
    idx_em = cluster(gm,x);
    t_em(i)=toc(t4);
    Acc_em(i)=accuracy(y,idx_em);

    t5=tic;
    idx_sc = spectralcluster(x,m);
    t_sc(i)=toc(t5);
    Acc_sc(i)=accuracy(y,idx_sc);

    t6=tic;
    idx_tensor = gmmtensor(x,m);
    t_td(i)=toc(t6);
    Acc_td(i)=accuracy(y,idx_tensor);
    
    t7=tic;
    [C,alg_idx]=scrlm(x,n,m,rho,F,i);
    Acc_scrlm(i)=accuracy(y,alg_idx.');
    t_scrlm(i)=toc(t7);

    t8=tic;
    [label_scrlm_kmeans,~]=scrlm_kmeans(x,n,m,rho,F,i);
    t_scrlm_kmeans(i)=toc(t8);
    Acc_scrlm_kmeans(i)=accuracy(y,label_scrlm_kmeans.');

    t9=tic;
    idx_robust_kmeans=robust_kmeans(x, k, m,delta, alpha);
    t_robust_kmeans(i)=toc(t9);
    Acc_robust_kmeans(i)=accuracy(y,idx_robust_kmeans.');

end

a1=mean(Acc_cl);
a2=mean(Acc_sc);
a3=mean(Acc_em);
a4=mean(Acc_td);
a5 = mean(Acc_tsne);
a6=mean(Acc_scrlm);
a7=mean(Acc_kmeans);
a8=mean(Acc_scrlm_kmeans);
a9=mean(Acc_robust_kmeans);

t11=mean(t_cl);
t22=mean(t_sc);
t33=mean(t_em);
t44=mean(t_td);
t55=mean(t_tsne);
t66=mean(t_scrlm);
t77=mean(t_kmeans);
t88=mean(t_scrlm_kmeans);
t99=mean(t_robust_kmeans);

T1=table(Acc_cl,Acc_sc,Acc_em,Acc_td,Acc_tsne,Acc_scrlm,Acc_kmeans,Acc_scrlm_kmeans,Acc_robust_kmeans,'VariableNames', {'CL','SC','EM','TD','tsne+kmeans++','SCRLM','kmeans++','scrlm+kmeans','robust kmeans++'});
writetable(T1, 'Acc_mnist.txt')

T2=table(t_cl,t_sc,t_em,t_td,t_tsne,t_scrlm,t_kmeans,t_scrlm_kmeans,t_robust_kmeans,'VariableNames', {'CL','SC','EM','TD','tsne+kmeans++','SCRLM','kmeans++','scrlm+kmeans','robust kmeans++'});
writetable(T2, 'Time_mnist.txt')

T3=table(a1,a2,a3,a4,a5,a6,a7,a8,a9,'VariableNames', {'CL','SC','EM','TD','tsne+kmeans++','SCRLM','kmeans++','scrlm+kmeans','robust kmeans++'});
writetable(T3, 'Acc_mnist_avg.txt')

T4=table(t11,t22,t33,t44,t55,t66,t77,t88,t99,'VariableNames', {'CL','SC','EM','TD','tsne+kmeans++','SCRLM','kmeans++','scrlm+kmeans','robust kmeans++'});
writetable(T4, 'Time_mnist_avg.txt')
