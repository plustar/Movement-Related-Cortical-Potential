clc;clear;
nfold=10;
nchn=11;
nsap=768;
Fstart=0.5;
Fend=10;
accuracy_strca=zeros(1,10,nchn,2);
for ncomp=1:nchn
    for sub=1:1
        disp([num2str(ncomp), ' ', num2str(sub)])
        train_ind=cell(7,1);test_ind=cell(7,1);dataf=cell(7,1);
        for mn=0:6
            % load shuffle index
            load(['Dataset/RandomIndex/index_sub',num2str(sub),'_mn',num2str(mn),'.mat'])
            train_ind{mn+1}=train_index;test_ind{mn+1}=test_index;
            % prepare dataloader
            dataf{mn+1}=zeros(nchn,nsap,size(train_index,2)+size(test_index,2));
            clear train_index test_index

            load(['Dataset/DataBank_Fourier/Motion',num2str(mn), ...
                '/Subject',num2str(sub),'/Fstart_0.5_Fend_10.mat'], 'data');
            dataf{mn+1}=permute(data, [2, 1, 3]);
        end

        % for each fold
        for n=1:nfold
            % split training set and testing set
            train_dataf=cell(7,1);test_dataf=cell(7,1);
            for mn=0:6
                train_dataf{mn+1}=dataf{mn+1}(:,:,train_ind{mn+1}(n,:));
                test_dataf{mn+1} =dataf{mn+1}(:,:,test_ind{mn+1}(n,:));
            end

            % spatial filter task-related component analysis
            S=zeros(nchn,nchn);Q=zeros(nchn,nchn);
            for mn=0:6
                [s,q]=sptrca_SQ(squeeze(train_dataf{mn+1}));
                S=S+s;Q=Q+q;
            end
            S=S/7;Q=Q/7;
            [V,D] = eig(S, Q, 'qz');
            [~,index]=sort(diag(D),'descend');
            w=V(:,index(1:ncomp));
            W=w.*reshape(sign(w(5,:)),1,ncomp);

            % templates for canonical correlation pattern
            mX=zeros(nsap,size(W,2),7);
            for mn=0:6
                mX(:,:,mn+1)=squeeze(mean(train_dataf{mn+1}, 3))'*squeeze(W);
            end

            % feature extraction of canonical correlation pattern
            train_dataff=[];test_dataff=[];train_label=[];test_label=[];
            for mn=0:6
                train_dataff=cat(3,train_dataff,train_dataf{mn+1});
                test_dataff =cat(3,test_dataff, test_dataf{mn+1});
                train_label=cat(1,train_label,mn*ones(size(train_dataf{mn+1},3),1));
                test_label=cat(1,test_label,mn*ones(size(test_dataf{mn+1},3),1));
            end

            train_data=Pattern_CCP(squeeze(train_dataff), mX,W);
            test_data =Pattern_CCP(squeeze(test_dataff) , mX,W);

            x_train = reshape(train_data, [], 21);
            x_test = reshape(test_data, [], 21);
            model=fitcecoc(x_train,train_label,'Learner','discriminant');
            pred_label=predict(model, x_test);
            accuracy_strca(sub,n, ncomp, 1)=mean(pred_label(:)==test_label(:));

            model=fitcecoc(x_train,train_label,'Learner','svm');
            pred_label=predict(model, x_test);
            accuracy_strca(sub,n, ncomp, 2)=mean(pred_label(:)==test_label(:));
        end
    end
end
save accuracy_strca.mat accuracy_strca
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
    [n_spl,~, n_cls]=size(mX);
    
    [n_chn, ~]=size(W);
    n_trl=size(X, 3);
    tmp_mX=mX-mean(mX,3);
    ru=zeros(n_trl, n_cls, 3);
    X=reshape(reshape(X,n_chn,[])'*W, [n_spl,n_trl,size(W,2)]);
    for nt=1:n_trl
        x=squeeze(X(:,nt,:))-mean(mX,3);
        for nc=1:n_cls
            ru(nt,nc, 1)=corr2(x, tmp_mX(:, :, nc));
%             disp(size(x))
%             disp(size(tmp_mX(:, :, nc)))
            [~, B]=canoncorr(x, tmp_mX(:, :, nc));
            ru(nt,nc, 2)=corr2(x*B, tmp_mX(:, :, nc)*B);
            tmpx1=x-tmp_mX(:, :, nc);
            
            tmpx2=squeeze(mean(tmp_mX(:,:,setdiff(1:n_cls, nc)), 3))-tmp_mX(:, :, nc);
            [A, ~]=canoncorr(tmpx1, tmpx2);
            ru(nt,nc, 3)=corr2(tmpx1*A, tmpx2*A);
        end
    end
end

function r=corr2_r1(a, b)
    a = a - mean2(a);
    b = b - mean2(b);
    r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
end

function r=corr2_r2(a, b)
    r = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b)));
end
