import shutil
import ultralytics
from ultralytics import YOLO
from os.path import join, exists
from os import makedirs
import logging


def gen_engine(verbose=False):
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
            if exists(engine_dest): 
                logging.debug(f'Found existing engine {engine_dest}. Skipping.')
                continue
            
            model = YOLO(model_path, verbose=False)
            logging.debug(f"Generating {dtype.upper()} model for: {model_name} ...")
            model.export(
                format="engine", 
                half=True if dtype == 'fp16' else False,
                int8=True if dtype == 'int8' else False,
                verbose=verbose
            )
            engine_path = join(model_folder, model_name + export_suffix)
            if not exists(engine_path): engine_path = model_name + export_suffix
            shutil.move(engine_path, engine_dest)
            logging.debug(f"Model generated at {engine_dest}")
    
    logging.info('Models generated: yolov5s, yolov10s, yolo-world.')
    logging.info(
        'Please run eval command with "--all" argument (recommended), or specify models with keywords \
(choose from "v5s", "v10s", "world").'
    )
    # TODO: cleanup
    
    
if __name__ == "__main__":
    gen_engine(verbose=False)
