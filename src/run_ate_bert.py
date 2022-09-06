from __future__ import absolute_import

from transformers import BertTokenizer, BertConfig
import tensorflow as tf
import numpy as np
from args.ate_args import parser
from reader.semeval_ate_reader import SemevalAteReader
from model.ate_bert import ATE_BERT
from utils.conlleval import tags_and_preds_iterator, return_report_by_iterator
from utils.utils import get_logger, get_output_dir, print_args, set_environ, set_seed
from utils.optimizer import learning_rate_linear_warmup_and_decay, train_op
import os


def train():
    pass

def evaluate(model, reader, data_generator, phrase, logger):
    logger.info(f"start evaluation: {phrase}")
    total_y_true, total_y_pred, total_loss = [], [], []
    for batch_dev_data in data_generator():
        return_dict = model(batch_dev_data)
        y_true, y_pred = reader.remove_pad(return_dict)
        loss = return_dict["loss"]
        total_y_true += y_true
        total_y_pred += y_pred
        total_loss.append(loss.numpy().tolist())
    eval_result = return_report_by_iterator(tags_and_preds_iterator(total_y_true, total_y_pred))
    eval_result = eval_result[:2]
    f1 = float(eval_result[1].strip().split()[-1])
    total_loss = np.mean(total_loss)
    for line in eval_result:
        logger.info(line.rstrip())
    logger.info(f"{phrase} loss: {total_loss:>9.6f}")
    logger.info(f"end evaluation: {phrase}")
    return f1

def main():
    args = parser.parse_args()
    set_environ(args.gpu_id)
    if args.data_seed is not None:
        set_seed(args.data_seed)
    output_path = get_output_dir(args)
    logger = get_logger(os.path.join(output_path, args.log_path, "0.log"))
    print_args(args, logger)

    bert_tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    if args.do_train:
        train_reader = SemevalAteReader(bert_tokenizer, args)
        train_data_generator, dev_data_generator = train_reader.data_generator(args.train_set, args.batch_size, shuffle=True, dev_split=0.1, few_count=args.few_count)
        step_per_epoch = train_reader.get_step_per_epoch()
        max_train_step = step_per_epoch * args.epoch
    if args.do_test:
        test_reader = SemevalAteReader(bert_tokenizer, args)
        test_data_generator = test_reader.data_generator(args.test_set, args.batch_size)

    if args.seed is not None:
        set_seed(args.seed)

    # bertconfig = BertConfig()
    model = ATE_BERT(args)
    if args.do_train:
        optimizer_1 = tf.keras.optimizers.Adam(learning_rate_linear_warmup_and_decay(args.bert_lr, args.warm_up_rate, max_train_step, args.bert_end_lr))
        optimizer_2 = tf.keras.optimizers.Adam(learning_rate_linear_warmup_and_decay(args.lr, args.warm_up_rate, max_train_step, args.end_lr))
    if args.ckpt is not None and args.ckpt != "None":
        checkpoint = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, os.path.join(output_path, args.ckpt), args.ckpt_save_count)

    max_dev_f1 = max_test_f1 = 0.0
    if args.do_train:
        logger.info("start training.")
        step = 0
        losses = []
        for epoch_iter in range(args.epoch):
            for batch_idx, batch_train_data in enumerate(train_data_generator()):
                with tf.GradientTape(persistent=True) as tape:
                    return_dict = model(batch_train_data, training=True)
                    loss = return_dict["loss"]
                    losses.append(loss.numpy())
                grads = tape.gradient(loss, model.variables)
                var_list = tape.watched_variables()
                grads_and_vars = [(grad, var) for grad, var in zip(grads, var_list) if grad != None]
                # optimizer_1.apply_gradients(grads_and_vars=grads_and_vars)
                train_op(grads_and_vars, optimizer_1, optimizer_2)

                step += 1
                if step % args.skep_step == 0:
                    logger.info(f"iteration: {epoch_iter+1} {(step-1)%step_per_epoch+1}/{step_per_epoch}, step: {step}, loss: {np.mean(losses):>9.6f}")
                    losses = []
                            
            if args.do_val and (epoch_iter + 1) % 30 == 0:
                dev_f1 = evaluate(model, train_reader, dev_data_generator, "dev", logger)                
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    logger.info(f"new best dev f1: {dev_f1}")
                    if args.ckpt is not None and args.ckpt != "None":
                        ckpt_manager.save(checkpoint_number=epoch_iter)
                        logger.info("model saved.")
                                    
            if args.do_test and (epoch_iter + 1) % 3 == 0:
                test_f1 = evaluate(model, test_reader, test_data_generator, "test", logger)
                if test_f1 > max_test_f1:
                    max_test_f1 = test_f1
                    logger.info(f"new best test f1: {test_f1}")
                                
        logger.info("end training.\n")
    if args.do_val:
        evaluate(model, train_reader, dev_data_generator, "dev", logger)
    if args.do_test:
        evaluate(model, test_reader, test_data_generator, "test", logger)
        
if __name__ == "__main__":
    main()