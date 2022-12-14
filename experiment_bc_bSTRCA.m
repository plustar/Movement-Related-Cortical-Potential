clc;clear;
addpath(genpath('external'))
nfold=10;
nchn=11;
nsap=768;
Fs = 256; N   = 8;
Fstart=0.5;
Fend=1:10;
accuracy=zeros(1,10,21);
for sub=1:1
    mn=0;f=10;
    for mn1=0:5
        for mn2=mn1+1:6
            mn=mn+1;
            % load random split index
            load(['Dataset/RandomIndex/index_sub',num2str(sub),'_mn',num2str(mn1),'.mat'])
            train_ind{1}=train_index;test_ind{1}=test_index;
            clear train_index test_index
            load(['Dataset/RandomIndex/index_sub',num2str(sub),'_mn',num2str(mn2),'.mat'])
            train_ind{2}=train_index;test_ind{2}=test_index;
            clear train_index test_index
            % load EEG data
            load(['Dataset/DataBank/Motion',num2str(mn1), ...
                '/Subject',num2str(sub),'/Fstart_0.5_Fend_',...
                num2str(Fend(f)),'.mat'], 'data');
            dataf1=permute(data, [2, 1, 3]);
            clear data
            load(['Dataset/DataBank/Motion',num2str(mn2), ...
                '/Subject',num2str(sub),'/Fstart_0.5_Fend_',...
                num2str(Fend(f)),'.mat'], 'data');
            dataf2=permute(data, [2, 1, 3]);
            clear data
            % for each fold
            for n=1:nfold
                disp(['sub',num2str(sub),'/pair',num2str(mn),'/fold',num2str(n)])
                % split training set and testing set
                train_dataf1=dataf1(:,:,train_ind{1}(n,:));
                train_dataf2=dataf2(:,:,train_ind{2}(n,:));
                test_dataf1=dataf1(:,:,test_ind{1}(n,:));
                test_dataf2=dataf2(:,:,test_ind{2}(n,:));
                % spatial filter task-related component analysis
                W1=sptrca(squeeze(train_dataf1));
                W2=sptrca(squeeze(train_dataf2));
                W=cat(2, W1, W2);
                % templates for canonical correlation pattern
                mX=zeros(nsap,size(W,2),2);
                mX(:,:,1)=squeeze(mean(train_dataf1, 3))'*squeeze(W);
                mX(:,:,2)=squeeze(mean(train_dataf2, 3))'*squeeze(W);
                % feature extraction of canonical correlation pattern
                train_dataf=cat(3,train_dataf1,train_dataf2);
                test_dataf=cat(3,test_dataf1,test_dataf2);
                train_label=ones(size(train_dataf,3),1);
                train_label(1:size(train_dataf1,3))=0;
                test_label=ones(size(test_dataf,3),1);
                test_label(1:size(test_dataf1,3))=0;

                train_data=Pattern_CCP(squeeze(train_dataf),squeeze(mX),squeeze(W));
                test_data =Pattern_CCP(squeeze(test_dataf),squeeze(mX),squeeze(W));
                
                model=fitcdiscr(train_data,train_label);
                pred_label=predict(model,test_data);
                accuracy(1,n,mn)=mean(pred_label(:)==test_label(:));
            end
        end
    end
end
save accuracy_strca.mat accuracy
function W = sptrca(eeg)
    [num_chans, num_smpls, num_trials]  = size(eeg);
    % Q
    UX = reshape(eeg, num_chans, num_smpls*num_trials);
    UX = bsxfun(@minus, UX, mean(UX,2));
    Q = UX*UX'/(num_smpls*num_trials);
    
    % S
    eeg=eeg-mean(eeg,2);
    U = squeeze(sum(eeg,3));
    V=zeros(num_chans,num_chans);
    for k=1:num_trials
        V = V + squeeze(eeg(:,:,k))*squeeze(eeg(:,:,k))';
    end
    S=(U*U'-V)/(num_smpls*num_trials*(num_trials-1));
    
    [V,D] = eig(S, Q, 'qz');
    [~,index]=sort(diag(D),'descend');
    
    W=V(:,index(1:2));
    W=W.*reshape(sign(W(5,:)),1,2);
end

function ru=Pattern_CCP(X, mX, W)
    %X: channel*sample*trial
    %mX: channel*sample*N_class
    %W: channel*n_fea
    %tmp_mX:sample*channel*N_class
    %tmp_X: sample*channel*trial
    [n_spl,~, n_cls]=size(mX);
    
    [n_chn, ~]=size(W);
    n_trl=size(X, 3);
    tmp_mX=mX;
    ru=zeros(n_trl, n_cls, 3);
    X=reshape(reshape(X,n_chn,[])'*W, [n_spl,n_trl,size(W,2)]);
    for nt=1:n_trl
        x=squeeze(X(:,nt,:));
        for nc=1:n_cls
            ru(nt, nc, 1)=corr2(x, tmp_mX(:, :, nc));
            
            [~, B]=canoncorr(x, tmp_mX(:, :, nc));
            ru(nt, nc, 2)=corr2(x*B, tmp_mX(:, :, nc)*B);
            tmpx1=x-tmp_mX(:, :, nc);
            tmpx2=squeeze(mean(tmp_mX(:,:,setdiff(1:n_cls, nc)), 3))-tmp_mX(:, :, nc);
            [A, ~]=canoncorr(tmpx1, tmpx2);
            ru(nt, nc, 3)=corr2(tmpx1*A, tmpx2*A);
        end
    end
    ru=reshape(ru, n_trl, n_cls*3);
end
