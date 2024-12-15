function d = l2_distance(X,Y)
    % 计算出x，y之间的欧式距离
    if (nargin < 2)
        [D N] = size(X);
        lengths = sum(X.^2,1);
        d = repmat(lengths,[N 1]) + repmat(lengths',[1 N]);
        d = d - 2* X'*X;
        
    else 
        XX = sum(X.^2,1);
        YY = sum(Y.^2,1);
        d = repmat(XX', [1 size(Y,2)]) + repmat(YY, [size(X,2) 1]);
        d = d - 2*X'*Y;
    
end
    