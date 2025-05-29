from util import parse_args, evaluate, get_model_path
import logging
from gen_engine_u import gen_engine

    
def main(args):
    if args.verbose is None: 
        args.verbose = False
        
    if args.command == 'gen': 
        gen_engine(verbose=args.verbose)
        logging.info('Models are successfully generated.')
        return
    
    # command == eval
    logging.info(f'Using dataset from {args.dataset}')
    if args.model is not None:
        args.all = False
    else:
        # TODO: v7
        logging.info(f'Running benchmarks for all models (yolov5s, yolov10s, yolo-world) with all data types (FP32, FP16, INT8)')
        args.all = True
        valid_models = [
            'models/yolov5su_fp32.engine',
            'models/yolov5su_fp16.engine',
            'models/yolov5su_int8.engine',
            'models/yolov10s_fp32.engine',
            'models/yolov10s_fp16.engine',
            'models/yolov10s_int8.engine',
            'models/yolov8s-worldv2_fp32.engine',
            'models/yolov8s-worldv2_fp16.engine',
            'models/yolov8s-worldv2_int8.engine',
        ]
    
    if not args.all:
        # de-duplication
        args.model = args.model.split(',')
        models = []
        versions = ['5', '10', 'world']
        for version in versions:
            for model_name in args.model:
                if version in model_name.lower():
                    models.append(model_name.lower())
                    break
        
        logging.debug('Running model name check and conversion ...')
        model_dir = './models'
        valid_models = []
        for model in models:
            logging.debug(f'Currenly checking: {model}')
            if '/' in model:
                # needs a complete path with suffix
                if not model.endswith('.engine'):
                    logging.debug('Invalid')
                    raise ValueError(
                        '''\
If you specify models with paths, please input the absolute or relative paths which are COMPLETE, \
namely with the file extension of ".engine".\nPlease check if the engine file exists. If not, you \
can try running "gen" command before evaluations, or specify model keywords (choose from "v5s", "v10s", \
"world") once you have appropriate engine files in the "models" folder or repository root.\
'''
                    )
                # should be /aaa/ccc.engine or aaa/ccc.engine
                logging.debug('Valid')
                valid_models.append(model)
            else:
                # aaa.engine or aaa, maybe in ./ or ./models
                logging.debug('Looking for engine files in repository root.')
                model_path_list = get_model_path(model, './')
                logging.debug(f'Found {len(model_path_list)} appropriate engine files.')
                if len(model_path_list) == 0:
                    logging.debug('No appropriate engine. Now looking for engine files in "models" folder.')
                    model_path_list = get_model_path(model, model_dir)
                    logging.debug(f'Finally found {len(model_path_list)} appropriate engine files.')
                if len(model_path_list) == 0:
                    raise FileNotFoundError(
                        'Could not find model engine files in "models" folder or repository root. Please confirm "--model" input.'
                    )
                valid_models += model_path_list
        logging.info(f'Valid models for evaluation: {", ".join(valid_models)}')
    
    for model in valid_models:
        logging.info(f'Evaluating model "{model}" ...')
        evaluate(model, args.dataset, args.verbose)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)