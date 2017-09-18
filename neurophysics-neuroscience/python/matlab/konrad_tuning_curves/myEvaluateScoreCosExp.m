function [ score ] = evaluateScoreCosExp(paras, spikes, angle)
%function [ score ] = evaluateScoreCosExp(paras, spikes, angle)
% returns a score (which will be minimized), takes as input the parameters, and also spikes and angles 

% for making the predictions, keep in mind that we can change the baseline
% (add a constant), scale the cosine, and shift the cosine
%so the predictions are some kind of a function of the parameters and the
%angle
    predictedF=?;
    %to score we can just calculate the log probability
    %we now need to calculate the probability or better logP (poisson
    %equation)
    
    %we want to maximize the log probability
    score=-sum(logP); % by default matlab will minimize
end

