## Dataset
Labeled dataset: Download DRC-D dataset from [this repository](https://github.com/nie-lang/RotationCorrection).

Unlabeled dataset: Download it from [Google Drive](https://drive.google.com/file/d/1I1ypsLm-TGbgWl59q_p4tLSM6sAYRvqV/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/11oclqmMQIK1GRUnmOipvFg) and then put it (training_unlabel) in the folder of DRC-D.


## Test 
The pre-trained models of rotation correction are available at [Google Drive](https://drive.google.com/file/d/1TFrz8zhsobh46SE3G5kMnvx9ksvZvhKP/view?usp=sharing) (for the supervised model) and [Google Drive](https://drive.google.com/file/d/1K6N5YJekPa6iEYqAreqWM1hY4nu1eO7Z/view?usp=sharing) (for the semi-supervised model).

Or you can download the pre-trained models from [Baidu Cloud](https://pan.baidu.com/s/1ucdJ2j3CYmOlBuVdgO8Gng) (for the supervised model) and [Baidu Cloud](https://pan.baidu.com/s/1Lj_yuoWXNJFWLsWCAyt7Yg) (for the semi-supervised model). Extraction code: 1234.


Modify the model_path and test_path in rotation/Codes/test.py and run:
```
python test.py
```

## Train
Modify the train_path, train_unlabel_path, and test_path in rotation/Codes/train.py and run:

```
python train.py
```

