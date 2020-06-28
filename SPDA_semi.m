function [W,P,S,Obj]  =SPDA_semi( LX,Y,X,h,m,alpha,maxiter)
% LX: labeled data, every column is a sample
% Y:  label vector
% X:  all train data  
% h:  k nearest neghborhood
% m:  final dimensions
% alpha: a parameter
% maxiter: the maximal numbre of iteration

%% load scale
[~,n] = size(X);  % n denotes the number of all samples
c = unique(Y);    
l = length(Y);    % l denotes the number of labeled samples
p0 = [];

%% initialization
% initialized P
for i=1:length(c)
    Xc{i} = LX(:,Y == i);  %  store the each class samples
    nc(i) = size(Xc{i},2);  %  record the number of each class
end
% H = eye(l) - 1/n*ones(l);
% St = LX*H*LX';  % St of labeled data
% invSt = inv(St);
for k = 1 : length(c)
    Xi = Xc{k};
    ni = nc(k);   
    distXi = L2_distance_1(Xi,Xi);
    [~, idx] = sort(distXi,2);
    S0{k} = construct_S0(idx, h, ni);   
    p0 = blkdiag(p0,S0{k});         
end
obj2 = zeros(1,length(c));
Obj = zeros(1,maxiter);
% initialized S
S0 = 1/(n-1)*ones(n,n)-diag(1/(n-1)*ones(1,n));

%% Iteration
for iter = 1:maxiter
p = [];
P = p0.^2;
S = S0.^2;

% Calculate lapalcian matrix L_p;
P = (P+P')/2;
D_p = diag(sum(P));
L_p = D_p - P;

% Calculate lapalcian matrix L_s;
S = (S+S')/2;
D_s = diag(sum(S));
L_s = D_s - S;

% Calculate projection matrix W
G = (LX*L_p*LX'+X*alpha*L_s*X'); 
[W,~,~] = eig1(G, m, 0, 0);
W = W*diag(1./sqrt(diag(W'*W)));

% Updata matrix S
distXx = L2_distance_1(W'*X,W'*X);
dis = 1./(distXx+diag(eps*ones(1,n)));
dis(logical(eye(size(dis)))) = 0; 
S = dis./repmat(sum(dis,2),1,n);
Obj1 = alpha*sum(sum(distXx.*(S.^2)));

% Updata matrix P
 for i = 1:length(c)
 Xc{i} = LX(:,Y==i); 
 nc(i) = size(Xc{i},2);
 Xi = Xc{i};
 ni = nc(i);
 distLXx = L2_distance_1(W'*Xi,W'*Xi);
 [dis, idx] = sort(distLXx,2);
 PP{i} = construct_S( dis+eps,idx, h, 2,ni);
 p = blkdiag(p,PP{i});
 obj2(i) = sum(sum((PP{i}.^2).*distLXx));
% obj2(i) = sum((sum (dis(:,2:h+1).^(1/(1-2)),2) ).^(1-2));
 end
 Obj2 = sum(obj2); 
 
% Change the variable
 p0 = p;
 S0 = S;
 
%  Objective function value
Obj(1,iter) = Obj1 + Obj2;

end
end

