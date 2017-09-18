function [LL, UU, like, YY] = fa_em(XX, KK, varargin)
% fa_em - ML factor analysis using EM: [L, P, like, Y] = fa_em(X, K)
%
%    [LL, UU, like, Y] = FA_EM(X, K, ...) finds the varimax maximum
%    likelihood Factor Analysis fit to the data in X [nObservables x
%    nObservations] using the EM algorithm with K latents.  It returns
%    the estimated loadings (LL), unique variances (UU),
%    log-likelihood after each iteration (like), and latent mean
%    estimates Y [K x nObservations].
%
%       'nIter' -       Maximum number of iterations of the EM algorithm.
%	'tol' -		Convergence tolerance: when relative change in
%			likelihood per step drops below
%			this threshold, iteration is stopped.

nIter = 1000;
tol = 1e-7;
init = [];

optlistassign(who, varargin);



% discover dimensions
[DD,NN] = size(XX);

% subtract mean
XX=bsxfun(@minus, XX, mean(XX, 2));

% precompute variances
XX2=XX*XX';
diagXX2=diag(XX2);

if (isempty(init))
  cX = cov(XX');
  scale = det(cX)^(1/DD);
  if scale < eps scale = 1; end
  LL = randn(DD,KK)*sqrt(scale/KK);
  UU = diag(cX);
else
  LL = init.loadings;
  UU = init.uniquenesses;
end

% latent prior
II = eye(KK);

if nargout > 2 | tol > 0
  like = zeros(1, nIter);
end


% run EM
for iIter = 1:nIter

  UUinv = diag(1./UU);
  UUinvLL = bsxfun(@rdivide,LL, UU);
  
  YYcov = inv(LL'*UUinvLL + II);
  YY = YYcov*UUinvLL'*XX;
  YY2 = NN*YYcov + YY*YY';

  if nargout > 2 | tol > 0
    XXprec = UUinv - UUinvLL*YYcov*UUinvLL';
    like(iIter) = 0.5*NN*(logdet(XXprec)) - 0.5*sum(sum(XXprec.*XX2));
  end  

  LL = XX*YY'/YY2;
  UU = (diagXX2 - sum((LL*YY).*XX, 2))/NN;

  if tol > 0 & iIter > 1
    if abs(diff(like(iIter-1:iIter))) < tol*diff(like([1,iIter]))
      like = like(1:iIter);
      break;
    end
  end
end


%% rotate latents to variance-ranked projection
[LL,vars] = eigs(LL*LL', KK);
LL = LL*sqrt(vars);

%% redo inference to be consistent with rotated latents
UUinv = diag(1./UU);
UUinvLL = bsxfun(@rdivide,LL, UU);
YYcov = inv(LL'*UUinvLL + II);
YY = YYcov*UUinvLL'*XX;



