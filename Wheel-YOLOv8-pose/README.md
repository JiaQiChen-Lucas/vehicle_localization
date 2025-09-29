# Installation
cuda version:
```text
(Wheel-YOLOv8-pose) lucas@cloud:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

create conda env:
```shell
conda create -n Wheel-YOLOv8-pose python=3.11.9
conda activate Wheel-YOLOv8-pose
```

install pytorch:
```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

verify：
```shell
(Wheel-YOLOv8-pose) lucas@cloud:~$ python
Python 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.__version__)
2.1.0+cu118
>>> print(torch.version.cuda)
11.8
>>> torch.cuda.is_available()
True
```

install ultralytics:
```shell
pip install ultralytics==8.2.15
```

# Dataset
Due to the inclusion of some private data in our dataset, it may not be open sourced yet. If you need the relevant dataset, please contact us at `chenjiaqi@njust.edu.cn` to obtain it.

# Train model
Please run the `train_model.py` in `ultralytics-8.2.15`

# Find the best performing model
Please run the `find_best_model.py` in `ultralytics-8.2.15`

# Validate model performance
Please run the `validate_model.py` in `ultralytics-8.2.15`

# Export model
Due to the poor performance of the dfl structure in NPU processing, it needs to be moved outside the model. Therefore, when exporting the model, we need to make some modifications to the model structure.

Firstly, modify the `REPADetectHead` and `REPAPoseHead` in `ultralytics-8.2.15/ultralytics/nn/Addmodules/REPAPoseHead.py`, switch the version of the `forward()` method.

Next, please run the `export_onnx.py` in `ultralytics-8.2.15`.


