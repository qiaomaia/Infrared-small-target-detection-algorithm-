function [L,S,obj,err,iter] = etrpca_tnn_lp(X, lambda, tenW,weightB ,p,opts)

% Solve the Tensor Robust Principal Component Analysis based on Weighted Tensor Schatten p-Norm problem by ADMM
%
% min_{L,S} ||L||_sp^p+lambda*||S||_1, s.t. X=L+S
%
% ---------------------------------------------
% Input:
%       X       -    d1*d2*d3 tensor
%       lambda  -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       L       -    d1*d2*d3 tensor
%       S       -    d1*d2*d3 tensor
%       obj     -    objective function value
%       err     -    residual
%       iter    -    number of iterations
%
%
%
% Written by Pu Zhang

tol = 1e-4;
max_iter = 500;
rho = 1.2;
mu = 1.6e-4;
max_mu = 1e10;
DEBUG = 1;


if ~exist('opts', 'var')
    opts = [];
end
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end
if isfield(opts, 'N');           N = opts.N;                  end
dim = size(X);
[n1,n2,n3]=size(X);
L = zeros(dim);
S = L;
Y = L;
weightTen = ones(dim);

for iter = 1 : max_iter
        preT = sum(S(:) > 0);
    Lk = L;
    Sk = S;


    % Update L
    [L,tnnL] = prox_tnn(-S+X-Y/mu,weightB/mu,p);%weightS是B的加权
    % Update S
    S = prox_l1(-L+X-Y/mu,weightTen*lambda/mu);
    weightTen = p./ (abs(S) + 0.01)./tenW;%T的加权
    
    
    dY = L+S-X;
    err = norm(dY(:))/norm(X(:));
    % Coverge condition
 %   chgL = max(abs(Lk(:)-L(:)));
%    chgS = max(abs(Sk(:)-S(:)));
%    chg = max([ chgL chgS max(abs(dY(:))) ]);
      
     
    if DEBUG
        if iter == 1 || mod(iter, 1) == 0            
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                   ', err=' num2str(err)...
                    ',|T|0 = ' num2str(sum(S(:) > 0))]); 
        end
    end
    currT = sum(Sk(:) > 0);
    if err < tol || (preT>0 && currT>0 && preT == currT)
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);
    
    disp(['The ' num2str(iter)  '-th Iteration']);
end


function N = rankN(X, ratioN)
    [~,~,n3] = size(X);
    D = Unfold(X,n3,1);
    [~, S, ~] = svd(D, 'econ');
    [desS, ~] = sort(diag(S), 'descend');
    ratioVec = desS / desS(1);
    idxArr = find(ratioVec < ratioN);
    if idxArr(1) > 1
        N = idxArr(1) - 1;
    else
        N = 1;
    end