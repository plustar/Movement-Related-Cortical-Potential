clc;clear;
nfold=10;
Fs = 256; N   = 8;
Fstart=0.5;Fend=1:10;
size_sub=1; size_chn=11;
for sub=1:1
    for fst=1:length(Fstart)
        for fen=1:length(Fend)
            disp(['sub_', num2str(sub),'_fstart_',num2str(fst),'_fend_',num2str(fen)])
            
            for mn=0:6
                if exist(['Dataset/DataBank/Motion',num2str(mn),'/Subject',num2str(sub),...
                    '/Fstart_',num2str(Fstart(fst)),'_Fend_',num2str(Fend(fen)),'.mat'],'file')
                    continue
                end
                if ~exist(['Dataset/DataBank/Motion',num2str(mn),'/Subject',num2str(sub)],'dir')
                    mkdir(['Dataset/DataBank/Motion',num2str(mn),'/Subject',num2str(sub)])
                end
                load(['Dataset/OData/ME_motion_',num2str(mn),...
                    '_sub', num2str(sub), '.mat'], 'data');
                data=zscore(data,0,1);
                data=cat(1,flipud(data), data);
                h  = fdesign.bandpass('N,F3dB1,F3dB2', N, Fstart(fst), Fend(fen), Fs);
                Hd = design(h, 'butter');
                tmpdata=filter(Hd, data, 1);
                data=tmpdata(size(data, 1)/2+1:end,:,:);

                save(['Dataset/DataBank/Motion',num2str(mn),'/Subject',num2str(sub),...
                    '/Fstart_',num2str(Fstart(fst)),'_Fend_',num2str(Fend(fen)),'.mat'],'data')
            end
        end
    end
end