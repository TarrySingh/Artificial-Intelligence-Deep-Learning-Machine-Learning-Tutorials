function seq = cosmoother_gpfa_viaOrth_fast(seq, params, mList)
%
% seq = cosmoother_gpfa_viaOrth_fast(seq, params, mList)
%
% Performs leave-neuron-out prediction for GPFA.  This version takes 
% advantage of R being diagonal for computational savings.
%
% INPUTS:
%
% seq         - test data structure
% params      - GPFA model parameters fit to training data
% mList       - number of top orthonormal latent coordinates to use for 
%               prediction (e.g., 1:5)
%
% OUTPUTS:
%
% seq         - test data structure with new fields ycsOrthXX, where XX are
%               elements of mList.  seq(n).ycsOrthXX has the same dimensions
%               as seq(n).y.
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu

  if ~params.notes.RforceDiagonal
    fprintf('ERROR: R must be diagonal to use cosmoother_gpfa_viaOrth_fast.\n');
    return
  end

  [yDim, xDim] = size(params.C);
  Rinv         = diag(1./diag(params.R));  
  CRinv        = params.C' * Rinv;
  CRinvC       = CRinv * params.C;

  [blah, Corth, TT] = orthogonalize(zeros(xDim, 1), params.C);
  
  Tall = [seq.T];
  Tu   = unique(Tall);
  
  for n = 1:length(seq)
    for m = mList
      fn          = sprintf('ycsOrth%02d', m);
      seq(n).(fn) = nan(yDim, seq(n).T);
    end
  end
  
  for j = 1:length(Tu)
    T     = Tu(j);
    Thalf = ceil(T/2);
    
    [K_big, K_big_inv, logdet_K_big] = make_K_big(params, T);
    
    K_big = sparse(K_big);
    
    blah      = cell(1, T);
    [blah{:}] = deal(CRinvC);    
    invM      = invPerSymm(K_big_inv + blkdiag(blah{:}), xDim,...
                           'offDiagSparse', true);
    
    % Process all trials with length T
    nList     = find(Tall == T);
    dif       = bsxfun(@minus, [seq(nList).y], params.d); % yDim x sum(T)
    CRinv_dif = CRinv * dif;                              % xDim x sum(T)
    
    for i = 1:yDim      
      % Downdate invM to remove contribution of neuron i
      ci_invM    = nan(Thalf, xDim*T);
      ci_invM_ci = nan(Thalf, T);      
      idx        = 1: xDim : (xDim*T + 1);
      ci         = params.C(i,:)' / sqrt(params.R(i,i));
      for t = 1:Thalf
        bIdx         = idx(t):idx(t+1)-1;
        ci_invM(t,:) = ci' * invM(bIdx,:);
      end
      for t = 1:T
        bIdx            = idx(t):idx(t+1)-1;
        ci_invM_ci(:,t) = ci_invM(:,bIdx) * ci;
      end      
      ci_invM = fillPerSymm(ci_invM, xDim, T, 'blkSizeVert', 1);    % T x (xDim*T)
      term    = (fillPerSymm(ci_invM_ci, 1, T) - eye(T)) \ ci_invM; % T x (xDim*T)
      % Note: fillPerSymm is not worth doing on following line because
      % the time it takes to compute bottom half of invM_mi (presumably 
      % super-optimized by Matlab since it's just matrix operations) is 
      % LESS than time it takes to copy elements over (presumably   
      % non-optimized).
      invM_mi = invM - ci_invM' * term;                             % (xDim*T) x (xDim*T)
      
      % Subtract out contribution of neuron i 
      CRinvC_mi = CRinvC - ci * ci';
      term1Mat  = reshape(CRinv_dif - params.C(i,:)' / params.R(i,i) * dif(i,:),... 
                          xDim*T, []); % (xDim*T) x length(nList)
      
      % Compute blkProd = CRinvC_big * invM efficiently
      % blkProd is block persymmetric, so just compute top half
      blkProd = zeros(xDim*Thalf, xDim*T);
      idx     = 1: xDim : (xDim*Thalf + 1);
      for t = 1:Thalf
        bIdx            = idx(t):idx(t+1)-1;
        blkProd(bIdx,:) = CRinvC_mi * invM_mi(bIdx,:);
      end
      blkProd = K_big(1:(xDim*Thalf), :) *...
                  fillPerSymm(speye(xDim*Thalf, xDim*T) - blkProd, xDim, T);
      xsmMat  = fillPerSymm(blkProd, xDim, T) * term1Mat; % (xDim*T) x length(nList)
            
      ctr = 1;
      for n = nList
        xorth = TT * reshape(xsmMat(:,ctr), xDim, T);
       
        for m = mList
          fn               = sprintf('ycsOrth%02d', m);
          seq(n).(fn)(i,:) = Corth(i,1:m) * xorth(1:m,:) + params.d(i);
        end
        
        ctr = ctr + 1;
      end
    end
    fprintf('Cross-validation complete for %3d of %d trial lengths\r', j, length(Tu));
  end
  fprintf('\n');
