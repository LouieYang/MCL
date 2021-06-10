# Complete Codes for Mutual Centralized Learning

We have provided the core algorithm of our method to calculate episodic loss during training in folder "../core\_codes" and provide complete code here for training and evaluations.

## Dataset prepare

The mini-/tieredimagenet data should be placed in dir "./data/miniImagenet" ("./data/tieredimagenet") with folder format

```
code
├── data
│   ├── miniImagenet
│   │   ├── train
│   │   │   ├──n01532829
│   │   │   ├──────n0153282900000987.png
│   │   ├── val
│   │   │   ├──
│   │   │   ├──────
│   │   ├── test
│   │   │   ├── 
│   │   │   ├──────
```

## Fast train and test

Example 1: MCL-Katz ResNet12 VanillaFCN GPU 0

```
sh ./fast_train_test.sh configs/miniImagenet/MEL_N5K1_R12_katz.yaml 0
```

Example 2: MCL ResNet12 PyramidGrid GPU 0

```
sh ./fast_train_test.sh configs/miniImagenet/MEL_N5K1_R12_Grids.yaml 0
```

We provide several config files in dir "./configs/miniImagenet/" for example

