clc
clear all
close all
fclose all;



%% Get all training logs for a given run
target_dir = '.';
s = {};
nm = {};
d = dir(target_dir);
j = 1;
for i = 1:length(d)
    if  any(strfind(d(i).name,'.log'))
        s = [s; sprintf('%s\\%s', target_dir, d(i).name)];
        nm = [nm; d(i).name];
    end
end
%% Loop over training logs and acquire data
D_count = 0;
G_count = 0;
for i = 1:length(s)
    fname = s{i,1};
    fid = fopen(s{i,1},'r');
    % Prepare bookkeeping for sv0
    if any(strfind(s{i,1},'sv'))
        if any(strfind(s{i,1},'G_'))
            G_count = G_count +1;
        else
            D_count = D_count + 1;
        end
    end
    itr = [];
    val = [];
    j = 1;
    while ~feof(fid);
        line = fgets(fid);
        parsed = sscanf(line, '%d: %e');
        itr(j) = parsed(1);
        val(j) = parsed(2);
        j = j + 1;
    end
    s{i,2} = itr;
    s{i,3} = val;
    fclose(fid);
end

%% Plot SVs and losses
close all;
Gcc = hsv(G_count);
Dcc = hsv(D_count);
gi = 1;
di = 1;
li = 1;
legendG = {};
legendD = {};
legendL = {};
thresh=2; % wavelet denoising threshold
losses = {};
for i=1:length(s)
    if any(strfind(s{i,1},'D_loss_real.log')) || any(strfind(s{i,1},'D_loss_fake.log')) || any(strfind(s{i,1},'G_loss.log'))
        % Select colors
        if any(strfind(s{i,1},'D_loss_real.log'))
            color1 = [0.7,0.7,1.0];
            color2 = [0, 0, 1];
            dlr = {s{i,2}, s{i,3}, wden(s{i,3},'sqtwolog','s','mln', thresh, 'sym4'), color1, color2};
            losses = [losses; dlr];
        elseif any(strfind(s{i,1},'D_loss_fake.log'))
            color1 = [0.7,1.0,0.7];
            color2 = [0, 1, 0];
            dlf = {s{i,2},s{i,3} wden(s{i,3},'sqtwolog','s','mln', thresh, 'sym4'), color1, color2};
            losses = [losses; dlf];
        else % g loss
            color1 = [1.0, 0.7,0.7];
            color2 = [1, 0, 0];
            gl = {s{i,2},s{i,3}, wden(s{i,3},'sqtwolog','s','mln', thresh, 'sym4'), color1 color2};
            losses = [losses; gl];
        end
        figure(1); hold on;
        % Plot the unsmoothed losses; we'll plot the smoothed losses later
        plot(s{i,2},s{i,3},'color', color1, 'HandleVisibility','off');
        legendL = [legendL; nm{i}];
        continue
    end
    if any(strfind(s{i,1},'G_'))
        legendG = [legendG; nm{i}];
        figure(2); hold on;
        plot(s{i,2},s{i,3},'color',Gcc(gi,:),'linewidth',2);
        gi = gi+1;
    elseif any(strfind(s{i,1},'D_'))
        legendD = [legendD; nm{i}];
        figure(3); hold on;
        plot(s{i,2},s{i,3},'color',Dcc(di,:),'linewidth',2);
        di = di+1;
    else
        s{i,1} % Debug print to show the name of the log that was not processed.
    end
end
figure(1); 
% Plot the smoothed losses last
for i = 1:3
% plot(losses{i,1}, losses{i,2},'color', losses{i,4}, 'HandleVisibility','off');
plot(losses{i,1},losses{i,3},'color',losses{i,5});
end
legend(legendL, 'Interpreter', 'none'); title('Losses'); xlabel('Generator itr'); ylabel('loss'); axis([0, max(s{end,2}), -1, 4]);

figure(2); legend(legendG,'Interpreter','none'); title('Singular Values in G'); xlabel('Generator itr'); ylabel('SV0');
figure(3); legend(legendD, 'Interpreter', 'none'); title('Singular Values in D'); xlabel('Generator itr'); ylabel('SV0');
