clc;clear;
dataind=1:11;
for movement=[0,1,6]
    markind=13;
    for sub=1:1
        moverecorder=[];
        for run=1:10
            load(['RawData_MAT\ME_sub', num2str(sub), '_run', num2str(run),'.mat']);
            data=downsample(permute(data, [2,1,3]), 2);
            moverecorder=cat(3,moverecorder, data(:,:,typ==movement));
        end
        if movement == 6
            movemark=squeeze(moverecorder(:,markind,:));
            m=movemark-cat(1, zeros(1,60), movemark(1:end-1,:));
            timed=[256*2.5+1:256*5.5];
    %         m=m(256*2.5+1:256*5.5,:);
            m=m(timed,:);
            select_tr=std(sgolayfilt(m, 1, 31), [], 1)<0.02;
    %         data=moverecorder(256*2.5+1:256*5.5, dataind, select_tr);
    %         mark=movemark(256*2.5+1:256*5.5, select_tr);
            data=moverecorder(timed, dataind, select_tr);
            mark=movemark(timed, select_tr);
            save(['OData/ME_motion_',num2str(movement),'_sub', num2str(sub), '.mat'], 'data', 'locat', 'mark');

    %         m=m.*(m>0);
    %         select_tr=std(sgolayfilt(m, 1, 31), [], 1)>0.05;
        end
        if movement == 0
            movemark=squeeze(moverecorder(:,markind,:));
            m=movemark-cat(1, zeros(1,60), movemark(1:end-1,:));
            m=-m(512+1:128*10,:);
            m=m.*(m>0);
            select_tr=std(sgolayfilt(m, 1, 31), [], 1)>0.05;
            m=m(:, select_tr);
            m=sgolayfilt(m, 1, 31)./max(abs(sgolayfilt(m, 1, 31)));
            m1=m;
            [c,r]=find(m>=0.05);
            [~, tmpr]=unique(r, 'first');
            locat=c(tmpr);
            if sub==15
                tmpc=c(r==18);
                locat(18)=tmpc(8);
                tmpc=c(r==27);
                locat(27)=tmpc(15);
            end
            locat=locat+512;
            tmp_recorder=moverecorder(:, :, select_tr);
            mark=zeros(768, numel(locat));
            data=zeros(768, numel(dataind), numel(locat));
            for i =1:numel(locat)
                mark(:, i)=squeeze(tmp_recorder(locat(i)-512+1:locat(i)+256, markind, i));
                data(:,:, i)=tmp_recorder(locat(i)-512+1:locat(i)+256, dataind, i);
            end
            save(['OData/ME_motion_',num2str(movement),'_sub', num2str(sub), '.mat'], 'data', 'locat', 'mark');
            %         h=figure('Visible', 'off');
            %         for i=1:size(m,2)
            %             subplot(6,10,i);
            % %             tmp1=m(:, i)/10;
            %             plot(m(:, i));
            %             hold on;
            %             q=zeros(1, 768);
            %             q(locat(i))=1;
            %             plot(q, 'linewidth', 2.5)
            %             axis([0, inf, 0,1])
            %         end
            % % 	saveas(h,['images/deri_sub', num2str(sub),'.jpg']);
        end
        if movement == 1
            movemark=squeeze(moverecorder(:,markind,:));
            m=movemark-cat(1, zeros(1,60), movemark(1:end-1,:));
            m=m(512+1:128*10,:);
            m=m.*(m>0);
            select_tr=std(sgolayfilt(m, 1, 31), [], 1)>0.05;
            if sub==5
                select_tr(6)=0;
                select_tr(26)=0;
            end
            if sub==11
                select_tr(13)=0;
                select_tr(19)=0;
                select_tr(21)=0;
                select_tr(27)=0;
                select_tr(28)=0;
            end
            if sub==12
                select_tr(55)=0;
            end
            m=m(:, select_tr);
            n=sgolayfilt(m, 1, 31)./max(abs(sgolayfilt(m, 1, 31)));
            [c,r]=find(n>=0.05);
            [~, tmpr]=unique(r, 'first');
            locat=c(tmpr);
            if sub==10
                tmpc=c(r==5);
                locat(5)=tmpc(20);
                tmpc=c(r==22);
                locat(22)=tmpc(29);
                tmpc=c(r==27);
                locat(27)=tmpc(12);
                tmpc=c(r==41);
                locat(41)=tmpc(13);
                tmpc=c(r==52);
                locat(52)=tmpc(11);
                tmpc=c(r==53);
                locat(53)=tmpc(43);
            end
            if sub==11
                tmpc=c(r==17);
                locat(17)=tmpc(4);
                tmpc=c(r==19);
                locat(19)=tmpc(10);
                tmpc=c(r==27);
                locat(27)=tmpc(3);
                tmpc=c(r==40);
                locat(40)=tmpc(4);
                tmpc=c(r==55);
                locat(55)=tmpc(168);
            end
            locat=locat+512;
            tmp_recorder=moverecorder(:, :, select_tr);
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