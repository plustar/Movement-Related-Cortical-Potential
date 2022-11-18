clc;clear;
% CHANNELNAME={'F3', 'Fz','F4', 'FCz', 'C1', 'Cz', 'C2', 'CPz', 'P3','Pz','P4'};
CHANNEL=[1:2:5,15, 29:1:31, 45, 55:2:59, 84:90];
% CHANNEL=65:67;
% elbow flexion; hand Y 85
% elbow extension; hand Y 85
% supination; hand X 84
% pronation; hand X 84
% hand close; 89/90
% hand open; 89/90 
% rest; 
mkdir('RawData_MAT')
mkdir('OData')
for i=1:1
    for j=1:10
        if i<10
            sub=['0', num2str(i)];
        else
            sub=num2str(i);
        end
        gdfpath=['RawData_GDF/S', sub, '_ME/motorexecution_subject', num2str(i),'_run', num2str(j),'.gdf'];
%         gdfpath=['RawData_GDF/S', sub, '_MI/motorimagination_subject', num2str(i),'_run', num2str(j),'.gdf'];
        matpath=['RawData_MAT/ME_sub', num2str(i),'_run', num2str(j),'.mat'];
%         if exist(matpath, 'file')
%             continue
%         end
        disp(matpath)
        [s, h]=sload(gdfpath);
        s=s(:, CHANNEL)';
        pos1=h.EVENT.POS(1:4:168);
        pos2=h.EVENT.POS(4:4:168);
        typ=h.EVENT.TYP(4:4:168)-1536;
        data=zeros(size(CHANNEL, 2), h.SampleRate*10, size(h.TRIG, 1));
        for k=1:size(data, 3)
            data(:, :, k)=s(:, pos1(k):pos1(k)-1+h.SampleRate*10);
        end
        save(matpath, 'data', 'typ');
    end
end
