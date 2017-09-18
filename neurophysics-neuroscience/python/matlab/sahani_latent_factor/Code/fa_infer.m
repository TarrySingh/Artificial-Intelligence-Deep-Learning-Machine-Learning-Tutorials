function [YY] = fa_infer(XX, LL, UU)
% fa_em - ML factor analysis using EM: [L, P, like, Y] = fa_em(X, K)
%
%    [LL, UU, like, Y] = FA_EM(X, K, ...) finds the varimax maximum
%    likelihood Factor Analysis fit to the data in X [nObservables x
%    nObservations] using the EM algorithm with K latents.  It returns
%    the estimated loadings (LL), unique variances (UU),
%    log-likelihood after each iteration (like), and latent mean
%    estimates Y [K x nObservations].


% discover dimensions
[DD,KK] = size(LL);
II = eye(KK);

UUinvLL = bsxfun(@rdivide,LL,UU);
YYcov = inv(LL'*UUinvLL + II);
YY = YYcov*UUinvLL'*XX;
