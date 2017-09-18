function [ score ] = evaluateScoreCosExp(paras, spikes, angle)
    predictedF=exp(paras(1)+paras(2)*cos(angle-paras(3)));
    logP=spikes.*log(predictedF)-predictedF-log(factorial(spikes));
    score=-sum(logP);
end

