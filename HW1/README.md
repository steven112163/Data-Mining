# Data-Mining HW1

## Prerequisites
* pandas >= 1.1.4
* scikit-learn >= 0.23.2
* matplotlib >= 3.3.2
* xlrd >= 1.2.0

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
|-m, --mode|0: cross validation, 1: prediction|0 (0-1)|  
```shell script
$ python3 classifier.py [-m (0-1)]
```
