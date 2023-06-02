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

8.Run "experiment_bc_bFBTRCA.m" or other experiment files. The code adpated for the nested cross validation will be updated soon...

Please cite the following paper if you use the codes:

[1] Feng Duan, Hao Jia, Zhe Sun, et al. Decoding Premovement Patterns with Task-Related Component Analysis. Cognitive Computation, 2021. https://doi.org/10.1007/s12559-021-09941-7

[2] Feng Duan, Hao Jia, Zhiwen Zhang, et al. On the robustness of EEG tensor completion methods. Science China Technological Sciences, 2021. https://doi.org/10.1007/s11431-020-1839-5

[3] Hao Jia, Zhe Sun, Feng Duan, et al. Improving Pre-movement Pattern Detection with Filter Bank Selection. Journal of Neural Engineering, 2022. https://doi.org/10.1088/1741-2552/ac9e75

[4] Hao Jia, Fan Feng, Feng Duan, et al. Towards Multi-Class Pre-Movement Classification. IEEE Journal of Biomedical and Health Informatics, 2023. http://doi.org/10.1109/JBHI.2023.3278747

[5] Hao Jia, Cesar F. Caiafa, Duan Feng, et al. Enabling Temporal-Spectral Decoding in Pre-movement Detection. 
https://doi.org/10.48550/arXiv.2212.09304
