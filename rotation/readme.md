## Dataset
Labeled dataset: Download DRC-D dataset from [this repository](https://github.com/nie-lang/RotationCorrection).

Unlabeled dataset: Download it from [Google Drive]() or [Baidu Cloud]().


## Test 
The pre-trained models of rotation correction are available at [Google Drive](https://drive.google.com/file/d/1TFrz8zhsobh46SE3G5kMnvx9ksvZvhKP/view?usp=sharing) (for the supervised model) and [Google Drive](https://drive.google.com/file/d/1K6N5YJekPa6iEYqAreqWM1hY4nu1eO7Z/view?usp=sharing) (for the semi-supervised model).

Or you can download the pre-trained models from [Baidu Cloud]() (for the supervised model) and [Baidu Cloud]() (for the semi-supervised model).


Modify the model_path and test_path in rotation/Codes/test.py and run:
```
python test.py
```

## Train
Modify the train_path, train_unlabel_path, and test_path in rotation/Codes/train.py and run:

```
python train.py
```

