# Movement-Related-Cortical-Potential
##Official Code for STRCA, FBTRCA for Movement-Related Cortical Potential in the Brain-Computer Interface context

##This is example codes of STRCA, FBTRCA, TTSNet, tested on the dataset that the movement onsets have been located and aligned.

Steps:

1.Download data from http://bnci-horizon-2020.eu/database/data-sets, 25. Upper limb movement decoding from EEG (001-2017); and extract files to the path "Dataset/RawData_GDF".

2.Run "gdf2mat.m" and convert GDF files to MAT files, Biosig Toolbox is necessary in this step.

3.Run "find_onset.m", "find_onset2.m", "find_onset3.m", "find_onset4.m", "find_onset5.m", which locate the movement onset and extract the EEG signals in readiness potential and movement-monitoring potential.

4.Run "analysis_0_randomindex.m" to split the dataset into the training set and testing set.

5.Run "analysis_1_filterbank.m" to divide the EEG signals into the low-frequency filter banks.

6.Run "experiment_bc_bSTRCA.m".

7.Download the codes for feature selection method from 

  (1) https://www.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library
  
  (2) https://github.com/sanjitsbatra/Minimal-Complexity-Machines-Feature-Selection/tree/master/KDD%20Code
  
  and extract files to the dir "external".

Notes: The second link to feature selection method contains the mRMR method and the code for quantile discretization. However, the memory increases when the mRMR method repeats in matlab code panel. When using the mRMR in this link many times, it is necessary to exit the current matlab programme and restart the panel. The alternate choice of mRMR is the first link.

8.Run "experiment_bc_bFBTRCA.m" or other experiment files.

Please cite the following paper if you use the codes:

[1] Feng Duan, Hao Jia, Zhe Sun, et al. Decoding Premovement Patterns with Task-Related Component Analysis. Cogn Comput 13, 1389–1405 (2021). https://doi.org/10.1007/s12559-021-09941-7

[2] Hao Jia, Zhe Sun, Feng Duan, et al. Improving Pre-movement Pattern Detection with Filter Bank Selection. J. Neural Eng. 19 066012. https://doi.org/10.1088/1741-2552/ac9e75

[3] Hao Jia, Fan Feng, Feng Duan, et al. Towards Multi-Class Pre-Movement Classification. arXiv, October 6, 2022. http://arxiv.org/abs/2201.12157.


[4] Hao Jia, Cesar F. Caiafa, Duan Feng, et al. Enabling Temporal-Spectral Decoding in Pre-movement Detection.
