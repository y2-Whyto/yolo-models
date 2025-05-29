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

> **Note:** Modify `data=` argument in `eval_u.py` to test on different datasets. It needs a very long time to download COCO2017 (with `coco.yaml`). If you just want to check if this program works, `coco8.yaml` or `coco128.yaml` is recommended.


