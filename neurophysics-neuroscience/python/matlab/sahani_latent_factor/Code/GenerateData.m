function [xx,yy,zz,CC] = GenerateData(varargin);

%% optional settings:
nTime = 50;
nTrial = 100;
aligned = 1;

optlistassign(who, varargin);

%% these can't be changed
nLatent = 4;
nNeuron = 100;


%% make the data predictable
rng default;
rng(0);

yy = zeros(nLatent, nTime, nTrial);


tt = [0:nTime-1]./nTime;

%% Latents

if (aligned)
  lat12phase = zeros([1, 1, nTrial]);
  lat34phase = zeros([1, 1, nTrial]);
else
  lat12phase = rand([1, 1, nTrial]);
  lat34phase = rand([1, 1, nTrial]);
end  

yy(1, :, :) = sin(2*pi*2*(bsxfun(@plus, tt, lat12phase)));
yy(2, :, :) = cos(2*pi*3*(bsxfun(@plus, tt, lat12phase)));
yy(3, :, :) = sin(2*pi*4*(bsxfun(@plus, tt, lat34phase)));
yy(4, :, :) = cos(2*pi*5*(bsxfun(@plus, tt, lat34phase)));


%% Loadings

CC = zeros(nNeuron, nLatent);
CC(1:30, 1:2) = [linspace(0,10,30); linspace(10,0,30)]';
CC(31:70, :) =  [linspace(0,5,40); linspace(5,0,40); ...
                 linspace(0,5,40); linspace(5,0,40)]';
CC(71:100, 3:4) = [linspace(0,10,30); linspace(10,0,30)]';

%% Mean

mu(1:25,1) = 30;
mu(26:50,1) = 10;
mu(51:75,1) = 0;
mu(76:100,1) = 5;

zz = bsxfun(@plus, mu, CC*reshape(yy, nLatent, []));

xx = reshape(poissrnd(log(1+exp(zz))/50), [nNeuron, nTime, nTrial]);



