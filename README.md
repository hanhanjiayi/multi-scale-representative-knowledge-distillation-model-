# Continual Multiple Adverse Weather Image Restoration Based on Multi-Scale Representative Knowledge Distillation



## Requirements
- Python 3.6+  
```pip install -r requirements.txt```

## Experimental Setup
Our code requires three training datasets: OTS, Rain100H, Snow100K. Additionally, the test set used in our experiments is:SOTS-outdoor, Rain100H, Snow100K-M.
### Dataset
We recommend putting all datasets under the same folder (say $datasets) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure should look like:

```
$datasets/
|–– RESIDE/
    |–– OTS_beta/
        |–– hazy/
        |–– clear/
    |–– SOTS/
        |–– outdoor/
            |–– hazy/
            |–– clear/
|–– Rain100H/
    |–– train
        |–– rain
        |–– norain
    |–– test
        |–– rain
        |–– norain
|–– Snow100K
    |–– train
        |–– synthetic
        |–– gt
    |–– test
        |–– Snow100K-M
            |–– synthetic
            |–– gt
```



## Usage
If you want to test the results mentioned in our paper, run
```
python test.py --task_order haze rain snow --exp_name haze_rain_snow  --device cuda:0

```


