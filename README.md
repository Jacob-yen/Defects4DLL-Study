## Defects4DLL
This is the repository for our TSE paper "Evaluating Spectrum-based Fault Localization on  Deep Learning Libraries". It contains the code and benchmark Defects4DLL.

## Setup
Our experiments were conducted using Python 3.11 within two Docker containers: CUDA-10.2 (for PyTorch) and CUDA-11.2.2 (for PyTorch and TensorFlow). Instructions for setting up the environment are provided below.

### CUDA-10.2 (PyTorch)
#### Step1: Pull Docker.
```bash
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
```

#### Step2: Run Docker.
```bash
docker run --gpus=all -it -v <your-data-path>:/data --name="defects4dll-torch-cuda10" pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel  /bin/bash
```
As nvidia/cuda no longer provides the CUDA-10.2 image, we use the PyTorch image with CUDA-10.2.
Please replace `<your-data-path>` with the path to your data directory. This directory will be mounted to `/data` inside the Docker container to support data loading.

Download vim, wget, and tmux
```bash
apt-get update
apt-get install vim tmux wget
```

#### Step3: Install cudnn dependencies.
The following commands should be executed inside the Docker container.
```bash
wget https://developer.download.nvidia.cn/compute/redist/cudnn/v7.6.5/cudnn-10.2-linux-x64-v7.6.5.32.tgz
tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz
cp cuda/include/cudnn*.h /usr/local/cuda/include/
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### Step4: Install Python dependencies.
The following commands should be executed inside the Docker container.
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```
The default installation path is `/root/anaconda3`. Download pre-configured python enviroments from [here](https://zenodo.org/records/15760824). Extract the files and copy the enviroment folders to `/root/anaconda3/envs`.

Take the `pytorch` environment as an example:
```bash
cd /root/anaconda3/envs
cp -r /<your-save-path>/py311 .
cp -r /<your-save-path>/pytorch-cuda10/torch-* .
cp -r /<your-save-path>/pytorch-cuda11/torch-* .
```
This will take a while as each environment is larger than 1GB. After copying, you can activate the environment using:
```bash
# This will list all available environments
conda info --envs 
# Activate the Python 3.11 environment for running the experiments 
conda activate py311 
```

### CUDA-11.2 (PyTorch and TensorFlow)
#### Step1: Pull Docker.
```bash
docker pull nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
```

#### Step2: Run Docker.
For PyTorch
```bash
docker run --gpus=all -it -v <your-data-path>:/data --name="defects4dll-torch-cuda11" nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04  /bin/bash
```
For TensorFlow
```bash
docker run --gpus=all -it -v <your-data-path>:/data --
name="defects4dll-tf-cuda11" nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04  /bin/bash
```
Please replace `<your-data-path>` with the path to your data directory. This directory will be mounted to `/data` inside the Docker container to support data loading. 

As the image has already installed CUDA-11.2 and cudnn, we do not need to install them again. Directly follow the step-4 above to install Python dependencies.

Download vim, wget, and tmux
```bash
apt-get update
apt-get install vim tmux wget
```

## Quick Start
To reproduce all the bugs in our benchmark, you can run the following command:
```bash
python -u data/environments/execute-all.py cuda11 tensorflow
```

## Experiments

### Step1: Set up experiment configuration
The experiment configuration is defined in `src/config`. You can modify the file to configure the experiment settings.

```config
[general]
root_result_dir = /data/all-rebug-test/result_20250624-rule-based-900s
time_limit_seconds = 900
max_test_case_count = 10000000

[hybrid]
hybrid_techniques = rule
mutator_selection = mcmc
mutation_levels = api, control_flow, graph, variable
max_mutation_order = 5
model_name = gpt-3.5-turbo-0125
api_url = https://xxxx
api_key = xxx
temperature = 0.8

[compile]
source_compile_dir = /data/source_batch
bazel_path = /data/download
tf_third_party_path = /root/.cache/bazel/_bazel_root

[interpreter]
conda_path = /root/anaconda3/
```

### Step2: Construct running scripts
```bash
python -u src/command.py cuda11 tensorflow > run-tf-cuda11.sh
```
Random seed can be changed in command.py, and the default value is `20240222`.

### Step3: Execute the running scripts
```bash
bash run-tf-cuda11.sh
```

### Step4: Get localization results
After the experiments are completed, you can analyze the results using the provided scripts in `src/analysis`. For example, to analyze the results of the TensorFlow experiments, you can run:
```shell
python -u src/analysis/src/analysis/analysis_by_time.py --root_result_dir /data/all-rebug-test/result_20250624-rule-based-900s  --bug_types crash  --metrics mean_first_rank --formulas ochiai --save_dir results/tensorflow/result_20250624-rule-based-900s
```
The localization results will be saved in the specified `save_dir`. You can find the results for all three granularities (block, function and file) in the excel file.

## NOTE

We currently only release the data and code related to our mutation-based approaches. The data for the developer-provided tests depends on snapshots of the original repositories, totaling over 400GB across all 120 bugs. Due to storage limitations, we are unable to release this data publicly. If you are interested in accessing the developer-provided test data, please feel free to contact us.

## Citation
If you use this code or benchmark in your research, please cite our paper:
```bibtex
@ARTICLE{10930847,
  author={Yan, Ming and Chen, Junjie and Jiang, Tianjie and Jiang, Jiajun and Wang, Zan},
  journal={IEEE Transactions on Software Engineering}, 
  title={Evaluating Spectrum-Based Fault Localization on Deep Learning Libraries}, 
  year={2025},
  volume={51},
  number={5},
  pages={1399-1414},
  doi={10.1109/TSE.2025.3552622}}
```