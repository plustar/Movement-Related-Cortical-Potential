clc;clear;
nfold=10;
mkdir('Dataset/RandomIndex')
for sub=1:1
    for mn=0:6
        load(['Dataset/OData/ME_motion_',num2str(mn),'_sub',num2str(sub),'.mat'],'data')
        num_trial=size(data,3);
        [train_index, test_index]=random_slice(ones(num_trial,1), nfold);
        save(['Dataset/RandomIndex/index_sub',num2str(sub),'_mn',num2str(mn),'.mat'],'train_index','test_index')
    end
end

function [train_index, test_index]=random_slice(label, n)
    [C, ~, ic]=unique(label);
    class_num=numel(C);
    class_counts=accumarray(ic,1);
    class_counts=cat(1, zeros(1), class_counts);
    test_num=floor(class_counts/n);
    train_num=class_counts-test_num;
    test_index=zeros(n, sum(test_num(:)));
    train_index=zeros(n, sum(class_counts(:))-sum(test_num(:)));
    for c=1:class_num
        rng(c);
        tmp_perm=randperm(class_counts(c+1));
        for count_n=1:n
            %             disp(count_n)
            tmp=tmp_perm((count_n-1)*test_num(c+1)+1:count_n*test_num(c+1));
            %             disp(tmp)
            test_index(count_n, sum(test_num(1:c))+1:sum(test_num(1:c+1)))=sum(class_counts(1:c))+tmp;
            train_index(count_n, sum(train_num(1:c))+1:sum(train_num(1:c+1)))=sum(class_counts(1:c))+setdiff(1:class_counts(c+1), tmp);
        end
    end
end