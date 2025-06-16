<h1 align="center">Online Batch Selection Methods for Training Acceleration</h1>

## Environment
Create the environment for running our code:
```bash
conda create --name DivBS python=3.7.10
conda activate DivBS
pip install -r requirements.txt
```

## Data Preparation
For CIFAR datasets, the data will be automatically downloaded by the code. 

For Tiny-ImageNet, please download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it to the `_TINYIMAGENET` folder. Then, run the following command to prepare the data:
```bash
cd _TINYIMAGENET
python val_folder.py
```

## Running
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg cfg/cifar10_DivBS_01.yaml --wandb_not_upload 
```
The `--wandb_not_upload` is optional and is used to keep wandb log files locally without uploading them to the wandb cloud.
