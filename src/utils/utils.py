import os
import logging
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from fwd9m.tensorflow import enable_determinism


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def get_output_dir(args):
    root_dir = args.output_path
    while True:
        timestamp = int(time.time())
        local_time = time.localtime(timestamp)
        datetime = time.strftime("%Y-%m-%d_%H:%M:%S", local_time)
        cur_output_dir = os.path.join(root_dir, datetime)
        try:
            os.makedirs(cur_output_dir)
            break
        except:
            time.sleep(1)
    log_path = os.path.join(cur_output_dir, args.log_path)
    os.makedirs(log_path)
    if args.ckpt is not None and args.ckpt != "None":
        ckpt_path = os.path.join(cur_output_dir, args.ckpt)
        os.makedirs(ckpt_path)
    if args.do_pred:
        result_path = os.path.join(cur_output_dir, args.results)
        os.makedirs(result_path)
    return cur_output_dir

def print_args(args, logger):
    flags = {k: v for k, v in args.__dict__.items()}
    for k, v in flags.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))

def set_environ(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel("ERROR")
    # mixed_precision.set_global_policy("mixed_float16")

def set_seed(seed):
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    enable_determinism()

def find_masked_pos(label_ids):
    """
    Find the position of the unique [MASK] for prompt masked sequence.
    """
    mask_pos_id = -1
    for i, label in enumerate(label_ids):
        if label != -100:
            mask_pos_id = i
            break
    assert mask_pos_id != -1

    return mask_pos_id

def decode_prompt_predict(data):
    tokens, golden_labels = data["raw_tokens"], data["raw_labels"]
    samples = data["samples"]
    candidates = []
    res = []
    for span, logits in samples:
        if logits[1] > logits[0]:
            candidates.append([span, logits[1] - (span[1] - span[0])])
            # candidates.append([span, logits[1]])
    
    sorted(candidates, key=lambda x: x[1], reverse=True)
    while candidates != []:
        aspect = candidates.pop(0)
        res.append(aspect)
        idx = 0
        while idx < len(candidates):
            if set(range(candidates[idx][0][0], candidates[idx][0][1])) & set(range(aspect[0][0], aspect[0][1])) != set():
                candidates.pop(idx)
            else:
                idx += 1
        
    preds = [r[0] for r in res]
    pred_labels = ["O"] * len(golden_labels)
    for pred in preds:
        pred_labels[pred[0]] = "B-AS"
        for i in range(pred[0] + 1, pred[1]):
            pred_labels[i] = "I-AS"
    return golden_labels, pred_labels, tokens
