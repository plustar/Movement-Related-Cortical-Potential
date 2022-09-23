clc;clear;
dataind=1:11;
gaussEqn = 'a*exp(-((x-b)/c)^2)+d';
% NN=zeros()
for movement=[2]%[0,1,6]
    markind=13;
    for sub=1:1
        moverecorder=[];
        for run=1:10
            load(['RawData_MAT\ME_sub', num2str(sub), '_run', num2str(run),'.mat']);
            data=downsample(permute(data, [2,1,3]), 2);
            moverecorder=cat(3,moverecorder, data(:,:,typ==movement));
        end
        if movement == 2
            movemark=squeeze(moverecorder(:,markind-1,:));
            m=movemark-cat(1, zeros(1,60), movemark(1:end-1,:));
            m=m(512+1:128*8,:);
            select_tr=std(sgolayfilt(m, 1, 31), [], 1)>0;
            n=sgolayfilt(m, 1, 31)./max(abs(sgolayfilt(m, 1, 31)));
            nn=zeros(size(n));
            for i=1:size(n,2)
                sf=fit((1:512)',n(:,i),gaussEqn,'Start',[3,389,10,0]);
%                 disp([i,sf.a,sf.b,sf.c,sf.d])
                if abs(sf.a)<0.05 || sf.c>100 || sf.d>10
                    select_tr(i)=0;
                    continue
                end
                nn(:,i)=sf(1:512)-sf.d;
            end

            [c,r]=find(nn>=0.1);
            [tmpc, tmpr]=unique(r, 'first');
            locat=zeros(60,1);
            locat(tmpc)=c(tmpr);
            locat=locat+512;
            tmp_recorder=moverecorder(:, :, select_tr);
            locat=locat(select_tr);
            mark=zeros(768, numel(locat));
            data=zeros(768, numel(dataind), numel(locat));
            for i =1:numel(locat)
                mark(:, i)=squeeze(tmp_recorder(locat(i)-512+1:locat(i)+256, markind, i));
                data(:,:, i)=tmp_recorder(locat(i)-512+1:locat(i)+256, dataind, i);
            end
            save(['OData/ME_motion_',num2str(movement),'_sub', num2str(sub), '.mat'], 'data', 'locat', 'mark');
        end
    end
    % close all;
end