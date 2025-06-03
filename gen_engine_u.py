import shutil
import ultralytics
from ultralytics import YOLO
from os.path import join, exists
from os import makedirs
import logging


def gen_engine(data='coco8.yaml', verbose=False):
    model_folder = 'models'
    suffix = '.pt'
    export_suffix = '.engine'
    makedirs(model_folder, exist_ok=True)
    dtypes = ['fp32', 'fp16', 'int8']
    # TODO: yolov7
    for model_name in ['yolov5su', 'yolov10s', 'yolov8s-worldv2']:
        # check engine existence
        model_path = join(model_folder, model_name + suffix)
        if not exists(model_path): 
                model_path = model_name + suffix
        for dtype in dtypes:
            engine_dest = join(model_folder, model_name + '_' + dtype + export_suffix)
            if dtype != 'int8' and exists(engine_dest): 
                logging.debug(f'Found existing engine {engine_dest}. Skipping.')
                continue
            
            model = YOLO(model_path, verbose=False)
            logging.debug(f'Generating {dtype.upper()} model for: {model_name} ...')
            if dtype == 'int8':
                logging.debug(f'Calibration set for int8 model: {data}')
            model.export(
                format="engine", 
                half=(dtype == 'fp16'),
                int8=(dtype == 'int8'),
                data=data,
                dynamic=False,
                fraction=0.2,
                verbose=verbose,
            )
            engine_path = join(model_folder, model_name + export_suffix)
            if not exists(engine_path): engine_path = model_name + export_suffix
            shutil.move(engine_path, engine_dest)
            logging.debug(f'Model generated at {engine_dest}')
    
    logging.info('Models generated: yolov5s, yolov10s, yolo-world.')
    logging.info(
        'Please run eval command with "--all" argument (recommended), or specify models with keywords \
(choose from "v5s", "v10s", "world").'
    )
    # TODO: cleanup
    
    
if __name__ == "__main__":
    gen_engine(verbose=False)
