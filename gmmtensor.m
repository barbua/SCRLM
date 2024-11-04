function [Ind, mu, s_est]=gmmtensor(x,m)
tot=size(x,1);
d=size(x,2);
mu = zeros(d,1);
for i = 1:tot
    mu = mu + x(i, :)';
end
mu = mu / tot;
Sigma = zeros(d,d);
for i = 1:tot
    Sigma = Sigma + x(i, :)' * x(i, :);
end
Sigma = Sigma / tot;
[U,S,~] = svd(Sigma);
s_est = S(end,end);                                             % Estimate sigma^2
W = U(:,1:m) * sqrt(pinv(S(1:m,1:m)-diag(ones(m,1).*s_est)));   % Obtain whitening matrix
X_whit = x * W;                                                 % Whiten the data
TOL = 1e-2;                 % Convergence delta
maxiter = 1000;             % Maximum number of power step iterations
V_est = zeros(m,m);         % Estimated eigenvectors for tensor V
lambda = zeros(m,1);        % Estimated eigenvalues for tensor V
for i = 1:m
    % Generate initial random vector
    v_old = rand(m,1);
    v_old = v_old./norm(v_old);
    for iter = 1 : maxiter
        % Perform multilinear transformation
        v_new = (X_whit'* ((X_whit* v_old) .* (X_whit* v_old)))/tot;
        v_new = v_new - s_est * (W' * mu * dot((W*v_old),(W*v_old)));
        v_new = v_new - s_est * (2 * W' * W * v_old * ((W'*mu)' * (v_old)));
        % Defaltion
        if i > 1
            for j = 1:i-1
                v_new = v_new - (V_est(:,j) * (v_old'*V_est(:,j))^2)* lambda(j);
            end
        end
        % Compute new eigenvalue and eigenvector
        l = norm(v_new);
        v_new = v_new./norm(v_new);
        % Check for convergence
        if norm(v_old - v_new) < TOL
            V_est(:,i) = v_new;
            lambda(i,1) = l;
            break;
        end
        v_old = v_new;
    end
end
mu = pinv(W') * V_est * diag(lambda);

p=zeros(size(x,1),m);
for i=1:m
    p(:,i)=logpdf(x,mu(:,i).',s_est,d);
end
[M,Ind]= max(p,[],2);