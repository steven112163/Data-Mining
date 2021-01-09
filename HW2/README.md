# Data-Mining HW2



## Prerequisites
* pandas >= 1.1.4
* scikit-learn >= 0.23.2
* matplotlib >= 3.3.2
* xlrd >= 1.2.0
* scipy >= 1.5.4



## Technical Terms
|Symbol in File|Meaning in English|Meaning in Chinese|
|---|---|---|
|T|Temperature|體溫 °C|
|P|Pulse|心跳/分鐘|
|R|Respiration|呼吸/分鐘|
|NBPS|Non-invasive Blood Pressure (Systolic)|收縮壓 mmHg|
|NBPD|Non-invasive Blood Pressure (Diastolic)|舒張壓 mmHg|



## Run  
|Argument|Description|Default|
|---|---|---|
|-l, --learning_rate|Learning rate|0.1|
|-r, --regularization|0: without L2 regularization, 1: with L2 regularization|0 (0-1)|
|-p, --penalty|Hyperparameter of regularization|1.0|
|-m, --mode|0: cross validation, 1: prediction|0 (0-1)|
|-v, --verbosity|verbosity level|0 (0-1)|  
```shell script
$ python3 classifier.py [-l learning_rate] [-r (0-1)] [-p penalty] [-m (0-1)] [-v (0-1)]
```
