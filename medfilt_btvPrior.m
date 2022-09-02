function f = medfilt_btvPrior(x, imsize, P, alpha)

    if nargin < 3
        P = 1;
    end
    if nargin < 4
        alpha = 0.7;
    end
    
    % Reshape SR vector to image for further processing.
    X =vectorToImage(x, imsize);
    
    % Pad image at the border to perform shift operations.
    Xpad = padarray(X, [P P], 'symmetric');

    % Consider shifts in the interval [-P, +P].
    f = 0;
%     z = {};btv=zeros(size(X));
    for l = -P:P
        for m = -P:P
            if l ~= 0 || m ~= 0
                % Shift by l and m pixels.
                Xshift = Xpad((1+P-l):(end-P-l), (1+P-m):(end-P-m));
%                 z{l+P+1, m+P+1} = imageToVector( alpha^(abs(l) + abs(m)) * (Xshift - X) );
%                 z{l+P+1, m+P+1} =  alpha^(abs(l) + abs(m)) * (Xshift - X) ;
                f = f + alpha.^(abs(l)+abs(m)) .* sum( lossFun(Xshift(:) - X(:),imsize) ) ;%abs(Xshift(:) - X(:))
%                 btv=btv+z{l+P+1, m+P+1};
            end
        end
    end
    
   function h = lossFun(x,imsize)
    
    mu = 1e-4;
    x=imageToVector(medfilt2(vectorToImage(x, imsize),[3,3]));
    h = sqrt(x.^2 + mu); 
    