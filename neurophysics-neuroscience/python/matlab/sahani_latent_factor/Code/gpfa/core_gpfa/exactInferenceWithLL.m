function [seq, LL] = exactInferenceWithLL(seq, params, varargin)
%
% [seq, LL] = exactInferenceWithLL(seq, params,...)
%
% Extracts latent trajectories given GPFA model parameters.
%
% INPUTS:
%
% seq         - data structure, whose nth entry (corresponding to the nth 
%               experimental trial) has fields
%                 y (yDim x T) -- neural data
%                 T (1 x 1)    -- number of timesteps
% params      - GPFA model parameters 
%  
% OUTPUTS:
%
% seq         - data structure with new fields
%                 xsm (xDim x T)        -- posterior mean at each timepoint
%                 Vsm (xDim x xDim x T) -- posterior covariance at each timepoint
%                 VsmGP (T x T x xDim)  -- posterior covariance of each GP
% LL          - data log likelihood
%
% OPTIONAL ARGUMENTS:
%
% getLL       - logical that specifies whether to compute data log likelihood
%               (default: false)
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  getLL = false;
  assignopts(who, varargin);
  
  [yDim, xDim] = size(params.C);

  % Precomputations
  if params.notes.RforceDiagonal     
    Rinv     = diag(1./diag(params.R));
    logdet_R = sum(log(diag(params.R)));
  else
    Rinv     = inv(params.R);
    Rinv     = (Rinv+Rinv') / 2; % ensure symmetry
    logdet_R = logdet(params.R);
  end
  CRinv  = params.C' * Rinv;
  CRinvC = CRinv * params.C;
  
  Tall = [seq.T];
  Tu   = unique(Tall);
  LL   = 0;

  % Overview:
  % - Outer loop on each elt of Tu.
  % - For each elt of Tu, find all trials with that length.
  % - Do inference and LL computation for all those trials together.
  for j = 1:length(Tu)   
    T = Tu(j);

    [K_big, K_big_inv, logdet_K_big] = make_K_big(params, T);
    
    % There are three sparse matrices here: K_big, K_big_inv, and CRinvC_inv.
    % Choosing which one(s) to make sparse is tricky.  If all are sparse,
    % code slows down significantly.  Empirically, code runs fastest if
    % only K_big is made sparse.
    %
    % There are two problems with calling both K_big_inv and CRCinvC_big 
    % sparse:
    % 1) their sum is represented by Matlab as a sparse matrix and taking 
    %    its inverse is more costly than taking the inverse of the 
    %    corresponding full matrix.
    % 2) term2 has very few zero entries, but Matlab will represent it as a 
    %    sparse matrix.  This makes later computations with term2 ineffficient.
    
    K_big = sparse(K_big);

    blah        = cell(1, T);
    [blah{:}]   = deal(CRinvC);
    %CRinvC_big = blkdiag(blah{:});     % (xDim*T) x (xDim*T)
    [invM, logdet_M] = invPerSymm(K_big_inv + blkdiag(blah{:}), xDim,... 
    'offDiagSparse', true);
    
    % Note that posterior covariance does not depend on observations, 
    % so can compute once for all trials with same T.
    % xDim x xDim posterior covariance for each timepoint
    Vsm = nan(xDim, xDim, T);
    idx = 1: xDim : (xDim*T + 1);
    for t = 1:T
      cIdx       = idx(t):idx(t+1)-1;
      Vsm(:,:,t) = invM(cIdx, cIdx);
    end
    
    % T x T posterior covariance for each GP
    VsmGP = nan(T, T, xDim);
    idx   = 0 : xDim : (xDim*(T-1));
    for i = 1:xDim
      VsmGP(:,:,i) = invM(idx+i,idx+i);
    end
    
    % Process all trials with length T
    nList    = find(Tall == T);
    dif      = bsxfun(@minus, [seq(nList).y], params.d); % yDim x sum(T)
    term1Mat = reshape(CRinv * dif, xDim*T, []); % (xDim*T) x length(nList)

    % Compute blkProd = CRinvC_big * invM efficiently
    % blkProd is block persymmetric, so just compute top half
    Thalf   = ceil(T/2);
    blkProd = zeros(xDim*Thalf, xDim*T);
    idx     = 1: xDim : (xDim*Thalf + 1);
    for t = 1:Thalf
      bIdx            = idx(t):idx(t+1)-1;
      blkProd(bIdx,:) = CRinvC * invM(bIdx,:);
    end
    blkProd = K_big(1:(xDim*Thalf), :) *... 
                fillPerSymm(speye(xDim*Thalf, xDim*T) - blkProd, xDim, T);   
    xsmMat  = fillPerSymm(blkProd, xDim, T) * term1Mat; % (xDim*T) x length(nList)
    
    ctr = 1;
    for n = nList
      seq(n).xsm   = reshape(xsmMat(:,ctr), xDim, T);      
      seq(n).Vsm   = Vsm;
      seq(n).VsmGP = VsmGP;

      ctr = ctr + 1;
    end

    if getLL
      % Compute data likelihood
      val = -T * logdet_R - logdet_K_big - logdet_M -...
            yDim * T * log(2*pi);      
      LL  = LL + length(nList) * val - sum(sum((Rinv * dif) .* dif)) +...
            sum(sum((term1Mat' * invM) .* term1Mat'));
    end
  end

  if getLL
    LL = LL / 2;
  else 
    LL = NaN;
  end
