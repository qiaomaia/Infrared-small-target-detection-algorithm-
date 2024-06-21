function   x   =  solve_Lp_w( y, lambda, p )

% Modified by Dr. xie yuan
% lambda here presents the weights vector
J     =   2;  %2
mu = 2*1e-3;

% tau is generalized thresholding vector
tau   =  (2*lambda.*(1-p)).^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda).^((p-1)/(2-p));

x     =   zeros( size(y) );
% i0 is the number of zero elements after thresholding
i0    =   find( abs(y)>tau );

if  abs(y)>tau
    % lambda  =   lambda(i0);
    y0    =   y(i0);
    t     =   abs(y);
    lambda0 = lambda(i0);
    for  j  =  1 : J
        t    =  abs(y0) - p*lambda0.*(t).^(p-1);
    end
    x(i0)   =  sign(y0).*t;
end