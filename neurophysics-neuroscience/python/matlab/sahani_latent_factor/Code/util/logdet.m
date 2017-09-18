function D = logdet(A)
% logdet - fast robust log(det())
D = 2*sum(log(diag(chol(A))));

