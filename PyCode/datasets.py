import numpy as np
import h5py

class Dataset(object):
    def __init__(self, num_class, num_sample, databank='DataBank_Fourier') -> None:
        self.data=[]
        self.train_index=[]
        self.test_index=[]
        self.num_class = num_class
        self.sample_time = num_sample
        self.databank = databank

    
    def load_data(self, sub, fbank):
        self.fbank = list(fbank)
        for mn in range(self.num_class):
            index_file=h5py.File('Dataset/RandomIndex/index_sub'+\
                str(sub)+'_mn'+str(mn)+'.mat', 'r')
            
            self.train_index.append(index_file['train_index'])
            self.test_index.append(index_file['test_index'])
            tmpdata = np.zeros([len(index_file['train_index'])+len(index_file['test_index']),len(self.fbank),11,self.sample_time])
            for f in range(len(self.fbank)):
                data_file=h5py.File('Dataset/'+self.databank+'/Motion'+str(mn)+\
                    '/Subject'+str(sub)+'/Fstart_0.5_Fend_'+str(self.fbank[f])+'.mat', 'r')
                tmpdata[:,f,:,:]=data_file['data']
            self.data.append(tmpdata)

    def divide_data(self,n):
        train_data=np.zeros([1,len(self.fbank),11,self.sample_time])
        test_data=np.zeros([1,len(self.fbank),11,self.sample_time])
        train_label=np.zeros([1,1])
        test_label=np.zeros([1,1])
        for mn in range(self.num_class):
            train_data=np.concatenate([train_data, \
                self.data[mn][np.sort(self.train_index[mn][:,n]).astype (int)-1,:,:,:]], axis=0)
            test_data=np.concatenate([test_data, \
                self.data[mn][np.sort(self.test_index[mn][:,n]).astype (int)-1,:,:,:]], axis=0)
            train_label=np.concatenate([train_label,\
                np.ones([np.shape(self.train_index[mn])[0],1])*mn], axis=0)
            test_label=np.concatenate([test_label,\
                np.ones([np.shape(self.test_index[mn])[0],1])*mn], axis=0)
        train_data=train_data[1:,:,:,:]
        test_data=test_data[1:,:,:,:]
        train_label=train_label[1:]
        test_label=test_label[1:]
        return train_data, train_label, test_data, test_label