import torch
import numpy as np
from PyCode.datasets import Dataset
from PyCode.utils import train, test, torch_seed_initialize
from PyCode.models import FilterBankDepNet
from PyCode.strca import StandardTRCA
from scipy.io import savemat

import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
    torch_seed_initialize()
    lr=1e-3
    batch_size=50
    epochs=100
    nfold=10
    use_cuda = torch.cuda.is_available()
    accuracy_depthnet=np.zeros([1,nfold,10,epochs])
    for sub in range(1):
        dataset=Dataset(7, 768)
        dataset.load_data(sub+1, np.arange(10)+1)
        for n in range(nfold):
            print('subject:'+str(sub+1)+' nfold:'+str(n))
            X_train_tmp, y_train, X_test_tmp, y_test = dataset.divide_data(n)
            X_train = np.zeros((len(y_train), 10, 3, 768))
            X_test  = np.zeros((len(y_test), 10, 3, 768))
            strca=StandardTRCA(11,3,768, True)
            for f in np.arange(10):
                strca.fit(X_train_tmp[:,f,:,:], y_train)
                X_train[:,f,:,:]=strca.transform(X_train_tmp[:,f,:,:])
                X_test[:,f,:,:] =strca.transform(X_test_tmp[:,f,:,:])
            train_data=torch.Tensor(X_train)
            test_data=torch.Tensor(X_test)
            train_label=torch.Tensor(y_train).reshape(-1).long()
            test_label=torch.Tensor(y_test).reshape(-1).long()
            

            for iepc in range(10):
                torch_seed_initialize()
                model = FilterBankDepNet(num_fbanks=10, num_channels=3,num_samples=768,num_classes=7)
                loss_fn = torch.nn.CrossEntropyLoss()
                if use_cuda:
                    model = model.cuda()
                for fbank in range(10):
                    model.block1[fbank].load_state_dict(torch.load(\
                        'SavedModel/Sub'+str(sub)+'Fold'+str(n)+'Epoch'+str((iepc+1)*10)+'Fbank'+str(fbank)+'.pt'),strict=False)
                
                for name, param in model.block1.named_parameters():
                    param.requires_grad = False
                optim = torch.optim.Adam(model.block2.parameters(), lr=lr, weight_decay=0.1)
                train_container = TensorDataset(train_data, train_label)
                train_data_loader = DataLoader(
                    train_container, batch_size=batch_size, shuffle=True)
                test_container = TensorDataset(test_data, test_label)
                test_data_loader = DataLoader(test_container,batch_size=len(test_label))
                for epoch in tqdm(range(epochs),desc="Sub"+str(sub)+"Nfold"+str(n)):
                    train(model, use_cuda, train_data_loader, optim, loss_fn)
                    accuracy_depthnet[sub,n,iepc, epoch] = test(model, use_cuda, test_data_loader)
                    time.sleep(0.001)
                    pass

    savemat('accuracy_mc_ttsnet.mat',{"accuracy_ttsnet": accuracy_depthnet})
