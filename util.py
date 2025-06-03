import os
import argparse
import logging
from ultralytics import YOLO

class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter): 
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='ℹ️  Get YOLO models and run benchmarks.',
        add_help=False,
        formatter_class=Formatter,
    )
    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # for subcommand 'gen'
    gen_parser = subparsers.add_parser(
        'gen', 
        description='ℹ️  Run directly to check and generate models for benchmark.',
        help='Check and generate models for benchmark. Run this command first.',
        formatter_class=Formatter,
        add_help=False,
    )
    gen_parser.add_argument(
        '-d', '--data', '--dataset',
        dest='dataset',
        type=str,
        default='coco.yaml',
        help='Specify the calibration dataset for int8 models.',
    )
    gen_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose logs.'
    )
    gen_parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    
    # for subcommand 'eval'
    eval_parser = subparsers.add_parser(
        'eval', 
        description='ℹ️  Run benchmarks for YOLO models. Run this after runnning "gen".',
        help='Run benchmarks for YOLO models.', 
        formatter_class=Formatter,
        add_help=False,
    )
    eval_exclusive_group = eval_parser.add_mutually_exclusive_group()
    eval_exclusive_group.add_argument(
        '-a', '--all',
        action='store_false',  # False by defalut
        help='Run model benchmark for all models (yolov5s, yolo-world, yolov10s).\nGet deactivated when "--model" is specified.',
    )
    eval_exclusive_group.add_argument(
        '-m', '--model', '--models',
        dest='model',
        type=str,
        help='Specify the model(s) for benchmark. Recommend to choose from: yolov5s, yolo-world, yolov10s. Split with comma (",") if multiple models are specified.',
    )
    eval_parser.add_argument(
        '-d', '--data', '--dataset',
        dest='dataset',
        type=str,
        default='coco.yaml',
        help='Specify the name of the YAML file of the dataset to evaluate on.\nOnly Ultralytics detection datasets are supported. See https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/datasets',
    )
    eval_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose logs.',
    )
    eval_parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    
    args = parser.parse_args()
    
    # set logging
    logging_level = logging.DEBUG
    logging_level = logging.INFO
    logging_formatter = logging.Formatter(
        fmt='%(asctime)s %(message)s',
        datefmt='%H:%M:%S',
    )
    logging_fh = logging.FileHandler('run.log', 'w')
    logging_fh.setFormatter(logging_formatter)
    logging_ch = logging.StreamHandler()
    logging.basicConfig(
        level=logging_level,
        handlers=[logging_fh, logging_ch],
    )
    logging.debug('args:' + str(args))
    
    return args
    

def evaluate(model_path, data, verbose):
    model = YOLO(model_path, task='detect', verbose=verbose)
    results = model.val(data=data, verbose=verbose)
    print(results.box.map)

def get_model_path(model_name, model_dir):
    model_path_list = []
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.engine') and model_name in file_name:
            model_path_list.append(os.path.join(model_dir, file_name))
    return model_path_list
