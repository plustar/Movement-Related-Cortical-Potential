clc;clear;
addpath(genpath('Code'))
nfold=10;
nchn=11;
nsap=768;
Fstart=0.5;
Fend=1:10;
nbank=10;
accuracy_fbtrca=zeros(1,10,211);
for sub=1:1
    train_ind=cell(7,1);test_ind=cell(7,1);dataf=cell(7,1);
    for mn=0:6
        % load shuffle index
        load(['Dataset/RandomIndex/index_sub',num2str(sub),'_mn',num2str(mn),'.mat'])
        train_ind{mn+1}=train_index;test_ind{mn+1}=test_index;
        % prepare dataloader
        dataf{mn+1}=zeros(10,nchn,nsap,size(train_index,2)+size(test_index,2));
        clear train_index test_index
        for f=1:10
            load(['Dataset/DataBank_Fourier/Motion',num2str(mn), ...
                '/Subject',num2str(sub),'/Fstart_0.5_Fend_',...
                num2str(Fend(f)),'.mat'], 'data');
            dataf{mn+1}(f,:,:,:)=permute(data, [2, 1, 3]);
            clear data
        end
    end
    
    % for each fold
    for n=1:nfold
        disp(['sub',num2str(sub),'/fold',num2str(n)])
        % split training set and testing set
        train_dataf=cell(7,1);test_dataf=cell(7,1);
        for mn=0:6
            train_dataf{mn+1}=dataf{mn+1}(:,:,:,train_ind{mn+1}(n,:));
            test_dataf{mn+1} =dataf{mn+1}(:,:,:,test_ind{mn+1}(n,:));
        end
        % spatial filter task-related component analysis
        W=zeros(10,11,3);
        for f=1:10
            S=zeros(nchn,nchn);Q=zeros(nchn,nchn);
            for mn=0:6
                [s,q]=sptrca_SQ(squeeze(train_dataf{mn+1}(f,:,:,:)));
                S=S+s;Q=Q+q;
            end
            S=S/7;Q=Q/7;
            [V,D] = eig(S, Q, 'qz');
            [~,index]=sort(diag(D),'descend');

            w=V(:,index(1:3));
            W(f,:,:)=w.*reshape(sign(w(5,:)),1,3);
        end
        % templates for canonical correlation pattern
        mX=zeros(10,nsap,size(W,3),7);
        for f=1:10
            for mn=0:6
                mX(f,:,:,mn+1)=squeeze(mean(train_dataf{mn+1}(f,:,:,:), 4))'*squeeze(W(f,:,:));
            end
        end
        % feature extraction of canonical correlation pattern
        train_dataff=[];test_dataff=[];train_label=[];test_label=[];
        for mn=0:6
            train_dataff=cat(4,train_dataff,train_dataf{mn+1});
            test_dataff =cat(4,test_dataff, test_dataf{mn+1});
            train_label=cat(1,train_label,mn*ones(size(train_dataf{mn+1},4),1));
            test_label=cat(1,test_label,mn*ones(size(test_dataf{mn+1},4),1));
        end
        train_data=zeros(size(train_dataff,4),7,3,10); % number of trials, patterns, banks
        test_data=zeros(size(test_dataff,4),7,3,10); 
        for f=1:10
            train_data(:,:,:,f)=Pattern_CCP(squeeze(train_dataff(f,:,:,:)),...
                squeeze(mX(f,:,:,:)),squeeze(W(f,:,:)));
            test_data(:,:,:,f) =Pattern_CCP(squeeze(test_dataff(f,:,:,:)),...
                squeeze(mX(f,:,:,:)),squeeze(W(f,:,:)));
        end

        % feature selection with minimum redundancy maximum dependency
        train_data=reshape(train_data,size(train_data,1),210);
        test_data =reshape(test_data, size(test_data, 1),210);
        model=fitcecoc(squeeze(train_data),train_label,'Learner','svm');
        pred_label=predict(model,test_data);
        accuracy_fbtrca(sub, n,1)=mean(pred_label(:)==test_label(:));
            
        itrain_data=myQuantileDiscretize(train_data,5);
        seqsorted=mRMR(itrain_data, train_label, 210);
        for nfea=1:210
            model=fitcecoc(squeeze(train_data(:,seqsorted(1:nfea))),train_label,'Learner','svm');
            pred_label=predict(model,test_data(:,seqsorted(1:nfea)));
            accuracy_fbtrca(sub, n,nfea+1)=mean(pred_label(:)==test_label(:));
        end
    end
end
save accuracy_fbtrca.mat accuracy_fbtrca
function [S,Q] = sptrca_SQ(eeg)
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
end

function ru=Pattern_CCP(X, mX, W)
    %X: channel*sample*trial
    %mX: channel*sample*N_class
    %W: channel*n_fea
    %tmp_mX:sample*channel*N_class
    %tmp_X: sample*channel*trial
%     disp(size(X))
%     disp(size(mX))
%     disp(size(W))
    [n_spl,~, n_cls]=size(mX);
    
    [n_chn, ~]=size(W);
    n_trl=size(X, 3);
    tmp_mX=mX-mean(mX,3);
    ru=zeros(n_trl, n_cls, 3);
    X=reshape(reshape(X,n_chn,[])'*W, [n_spl,n_trl,size(W,2)]);
    for nt=1:n_trl
        x=squeeze(X(:,nt,:))-mean(mX,3);
        for nc=1:n_cls
%             disp(x)
            ru(nt,nc, 1)=corr2(x, tmp_mX(:, :, nc));
            
            [~, B]=canoncorr(x, tmp_mX(:, :, nc));
            ru(nt,nc, 2)=corr2(x*B, tmp_mX(:, :, nc)*B);
            tmpx1=x-tmp_mX(:, :, nc);
            
            tmpx2=squeeze(mean(tmp_mX(:,:,setdiff(1:n_cls, nc)), 3))-tmp_mX(:, :, nc);
            [A, ~]=canoncorr(tmpx1, tmpx2);
            ru(nt,nc, 3)=corr2(tmpx1*A, tmpx2*A);
        end
    end
end
