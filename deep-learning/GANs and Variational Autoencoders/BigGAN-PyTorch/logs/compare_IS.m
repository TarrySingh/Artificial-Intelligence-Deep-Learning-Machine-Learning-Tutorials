clc
clear all
close all
fclose all;



%% Get All logs and sort them
s = {};
d = dir();
j = 1;
for i = 1:length(d)
    if  any(strfind(d(i).name,'.jsonl'))
        s = [s; d(i).name];
    end
end


j = 1;
for i = 1:length(s)
    fname = s{i,1};
    % Check if the Inception metrics log exists, and if so, plot it
    [itr, IS, FID, t] = process_inception_log(fname(1:end - 10), 'log.jsonl');
    s{i,2} = itr;
    s{i,3} = IS;
    s{i,4} = FID;
    s{i,5} = max(IS);
    s{i,6} = min(FID);
    s{i,7} = t;
end
% Sort by Inception Score?
[IS_sorted, IS_index] = sort(cell2mat(s(:,5)));
% Cutoff inception scores below a certain value?
threshold = 22;
IS_index = IS_index(IS_sorted > threshold);

% Sort by FID?
[FID_sorted, FID_index] = sort(cell2mat(s(:,6)));
% Cutoff also based on IS?
% threshold = 0;
FID_index = FID_index(IS_sorted > threshold);



%% Plot things?
cc = hsv(length(IS_index));
legend1 = {};
legend2 = {};
make_axis=true;%false % Turn this on to see the axis out to 1e6 iterations
for i=1:length(IS_index)
    legend1 = [legend1; s{IS_index(i), 1}];
    figure(1)
    plot(s{IS_index(i),2}, s{IS_index(i),3}, 'color', cc(i,:),'linewidth',2)
    hold on;
    xlabel('itr'); ylabel('IS');
    grid on;
    if make_axis
        axis([0,1e6,0,80]); % 50% grid on;
    end
    legend(legend1,'Interpreter','none')
    %pause(1) % Turn this on to animate stuff
    legend2 = [legend2; s{IS_index(i), 1}];
    figure(2)
    plot(s{IS_index(i),2}, s{IS_index(i),4}, 'color', cc(i,:),'linewidth',2)
    hold on;
    xlabel('itr'); ylabel('FID');
    j = j + 1;
    grid on;
    if make_axis
        axis([0,1e6,0,50]);% grid on;
    end
    legend(legend2, 'Interpreter','none')
    
end

%% Quick script to plot IS versus timesteps
if 0
    figure(3);
    this_index=4;
    subplot(2,1,1);
    %plot(s{this_index, 2}(2:end), s{this_index, 7}(2:end) - s{this_index, 7}(1:end-1), 'r*');
    % xlabel('Iteration');ylabel('\Delta T')
    plot(s{this_index, 2}, s{this_index, 7}, 'r*');
    xlabel('Iteration');ylabel('T')
    subplot(2,1,2);
    plot(s{this_index, 2}, s{this_index, 3}, 'r', 'linewidth',2);
    xlabel('Iteration'), ylabel('Inception score')
    title(s{this_index,1})
end