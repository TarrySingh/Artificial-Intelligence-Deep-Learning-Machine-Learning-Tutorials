function [itr, IS, FID, t] = process_inception_log(fname, which_log)
f = sprintf('%s_%s',fname, which_log);%'G_loss.log');
fid = fopen(f,'r');
itr = [];
IS = [];
FID = [];
t = [];
i = 1;
while ~feof(fid);
    s = fgets(fid);
    parsed = sscanf(s,'{"itr": %d, "IS_mean": %f, "IS_std": %f, "FID": %f, "_stamp": %f}');
    itr(i) = parsed(1);
    IS(i) = parsed(2);
    FID(i) = parsed(4);
    t(i) = parsed(5);
    i = i + 1;
end
fclose(fid);
end