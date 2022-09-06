"""
args for ate task.
"""
from __future__ import absolute_import
import argparse
from utils.args import ArgumentGroup

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for finetuning.")
train_g.add_arg("lr", float, 1e-3, "Learning rate for training.")
train_g.add_arg("seed", int, 7, "Seed for all random operations. Non-determinism if set None.")
train_g.add_arg("dropout", float, 0.1, "Dropout rate.")
train_g.add_arg("warm_up_rate", float, 0.1, "Step rate for warm up.")
train_g.add_arg("end_lr", float, 1e-4, "Ending learning rate of decay.")
train_g.add_arg("bert_lr", float, 2e-5, "Learning rate of bert.")
train_g.add_arg("bert_end_lr", float, 2e-5, "Ending bert learning rate of decay.")

log_g = ArgumentGroup(parser, "loging", "logging related.")
log_g.add_arg("log_path", str, "log", "Path of log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("train_set", str, "/home/ycpan/semeval_data/processed/restaurants_16.train", "Path of train data.")
data_g.add_arg("dev_set", str, None, "Path of develop data.")
data_g.add_arg("test_set", str, "/home/ycpan/semeval_data/processed/restaurants_16.test", "Path of test data.")
data_g.add_arg("max_seq_len", int, 128, "Number of words of the longest sequence.")
data_g.add_arg("batch_size", int, 16, "Number of examples in a batch.")
data_g.add_arg("do_lower_case", bool, True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("label_schema", str, 'iob', "label schema")
data_g.add_arg("output_path", str, "output", "Path of output.")
data_g.add_arg("few_count", int, 64, "Count of raw data for few-shot learning, -1 for all data.")
data_g.add_arg("data_seed", int, 7, "Seed for first data shuffle.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("do_train", bool, True, "Whether to preform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to preform evaluation on dev dataset.")
run_type_g.add_arg("do_test", bool, True, "Whether to preform evaluation on test dataset.")
run_type_g.add_arg("do_pred", bool, False, "Whether to preform prediction on test dataset.")
run_type_g.add_arg("ckpt", str, None, "Path to save checkpoint. Not save if value is None.")
run_type_g.add_arg("ckpt_save_count", int, 1, "Max number of saved checkpoints. Save all checkpoints if value is None")
run_type_g.add_arg("skep_step", int, 10, "Skip train steps for logger info.")
run_type_g.add_arg("gpu_id", str, "3", "Ids of gpu to use. Split by ','.")
run_type_g.add_arg("pretrain_model_path", str, "/home/ycpan/transformers_model/bert-base-uncased", "Path of pretrain model.")
