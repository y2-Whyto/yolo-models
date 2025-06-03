## Environment Setup

```shell
conda install python=3.9 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ultralytics==8.3.107
pip install tensorrt==8.6.1 --extra-index-url https://pypi.nvidia.com
pip install onnx==1.17.0 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0 onnxslim
pip install 'numpy<2'
# git clone https://github.com/WongKinYiu/yolov7.git  # for yolov7
```

## Usage

+ Run `gen_engine_u.py` to download models and generate engine in different data types.
+ Run `eval_new.py` to run benchmarks. 

> **Note:** It needs a very long time to download COCO2017 (with `coco.yaml`).


# Notes

TensorRT download address:
- [For Ubuntu 22.04](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb)
- [For Ubuntu 20.04](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb)
- [For Ubuntu 18.04](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu1804-8.6.1-cuda-12.0_1.0-1_amd64.deb)
