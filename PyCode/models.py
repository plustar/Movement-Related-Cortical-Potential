import torch
import torch.nn as nn


class DepNet(nn.Module):
    def __init__(self,
                 num_samples: int = 768,
                 num_channels: int = 11,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(DepNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.ZeroPad2d([self.kernel_1//2-1,self.kernel_1//2,0,0]),
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1,
                      self.F1 * self.D, (self.num_channels, 1),
                      groups=self.F1,
                      bias=False), 
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), 
            nn.AvgPool2d((1, 4), stride=4), 
            nn.Dropout(p=dropout),
            nn.ZeroPad2d([self.kernel_2//2-1,self.kernel_2//2,0,0]),
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), 
            nn.ELU(), 
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )


        self.block2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.F1 * self.D*self.num_samples//32, num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x

# import torchsummary
# model=DepNet(768,11,num_classes=7).cuda()
# torchsummary.summary(model,input_size=(1,11,768))

class Classifier(nn.Module):
    def __init__(self, num_fbanks, num_classes):
        super(Classifier, self).__init__()
        self.block = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_fbanks*num_classes,num_fbanks*num_classes*2, bias=False),
            nn.ReLU(),
            nn.Linear(num_fbanks*num_classes*2,num_fbanks*num_classes//2, bias=False),
            nn.ReLU(),
            nn.Linear(num_fbanks*num_classes//2,num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x

class FilterBankDepNet(nn.Module):
    def __init__(self,
                num_fbanks: int=10,
                 num_samples: int = 768,
                 num_channels: int = 11,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(FilterBankDepNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.filterbanks = [nfbank for nfbank in range(num_fbanks)]

        self.block1 = nn.ModuleList([
            nn.Sequential(
                DepNet(
                    num_samples=self.num_samples,
                    num_channels=self.num_channels,
                    F1=self.F1,
                    F2=self.F2,
                    D=self.D,
                    num_classes=self.num_classes,
                    kernel_1=self.kernel_1,
                    kernel_2=self.kernel_2,
                    dropout=self.dropout,
                )
            ) for _ in self.filterbanks
            ])
        self.block2 = Classifier(num_fbanks, num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.block1[netindex](x[:,netindex,:,:].unsqueeze(1)).unsqueeze(2) for netindex in range(len(self.block1))], dim=2)
        x = self.block2(x)
        return x
