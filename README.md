# ssftt_test

Original implementation [paper](https://ieeexplore.ieee.org/document/9684381) | [git](https://github.com/zgr6010/HSI_SSFTT)

Validation SSFTT on pushbroom HS data

Data can be download from [here](https://storage.ai.ssau.ru/s/27ff2tYjf9nxEKx?path=%2Fdata%2Fdata%2Ftablet)

For training you must change parameters in script [Tablet_train.py](https://github.com/f-neumann77/ssftt_test/blob/main/Tablet_train.py) and after run it. 

For only predict you can use [SSFTT_Intro.ipynb](https://github.com/f-neumann77/ssftt_test/blob/main/SSFTT_Intro.ipynb)

### WARNING! vanilla dataloader has problem with memory. It will be fixed in later versions.

Results:

PCA for channels count: 250 -> 30


96.97 Kappa accuracy (%)

98.93 Overall accuracy (%)

97.46 Average accuracy (%)

[99.60 98.71 94.08] Each accuracy (%)

              precision    recall  f1-score   support

     class 1     0.9966    0.9960    0.9963    308732
     class 2     0.9387    0.9872    0.9624     40412
     class 3     0.9870    0.9408    0.9634     40855

    accuracy                         0.9893    389999
   macro avg     0.9741    0.9747    0.9740    389999
weighted avg     0.9896    0.9893    0.9894    389999

