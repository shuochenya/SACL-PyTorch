# SACL-PyTorch
Similarity-Agnostic Contrastive Learning with Alterable Self-Supervision

## Installation
- Python 3.8 
- PyTorch 1.13
- OpenCV

## Training
Run the following command to train a contrastive encoder with the similarity-agnostic regularization
```
python cl_train.py
```

## Test
Run the following command to evaluate the model by (fine-tuning) a linear classifier
```
python cl_test.py --model_path results/xxx.pth
```

A pretrained (regularized) model can be downloaded from [here](https://drive.google.com/file/d/1NkFd7C5mp2Hnx9Reh-1HWir6NihIUd9z/view)

This code is mainly inspired by [SimCLR](https://github.com/leftthomas/SimCLR)
