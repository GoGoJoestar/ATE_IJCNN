from __future__ import absolute_import

from transformers import BertTokenizer, BertConfig
import tensorflow as tf
import numpy as np
from args.ate_ss_args import parser
from reader.semeval_ate_ss_reader import SemevalAteSSReader
from model.ate_bert_ss import ATE_BERT_SS, ATE_BERT_SS_1, contrastive_loss_v2, contrastive_loss_v3, contrastive_loss_v4, contrastive_loss_v5, contrastive_loss_v5_ablation, kl_loss
from utils.conlleval import tags_and_preds_iterator, return_report_by_iterator
from utils.utils import get_logger, get_output_dir, print_args, set_environ, set_seed
from utils.optimizer import learning_rate_linear_warmup_and_decay, train_op
import os


def train():
    pass

def evaluate(model, reader, data_generator, phrase, logger, latest_memory, do_pred=False, result_path=None):
    logger.info(f"start evaluation: {phrase}")
    total_y_true, total_y_pred, total_loss = [], [], []
    for batch_data in data_generator():
        return_dict = model(batch_data, latest_memory)
        if do_pred:
            reader.save_predict(batch_data, return_dict, result_path)
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
    if args.seed is not None:
        set_seed(args.seed)
    output_path = get_output_dir(args)
    logger = get_logger(os.path.join(output_path, args.log_path, "0.log"))
    print_args(args, logger)

    bert_tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    if args.do_train:
        train_reader = SemevalAteSSReader(bert_tokenizer, args)
        train_data_generator, dev_data_generator = train_reader.data_generator(args.train_set, \
                                                                               args.train_dep_label_file, \
                                                                               args.train_pos_label_file, \
                                                                               args.train_dep_adj_file, \
                                                                               args.batch_size, \
                                                                               shuffle=True, \
                                                                               dev_split=0.1)
        step_per_epoch = train_reader.get_step_per_epoch()
        max_train_step = step_per_epoch * args.epoch
    if args.do_test:
        test_reader = SemevalAteSSReader(bert_tokenizer, args)
        test_data_generator = test_reader.data_generator(args.test_set, args.test_dep_label_file, args.test_pos_label_file, args.test_dep_adj_file, args.batch_size)

    # bertconfig = BertConfig()
    model = ATE_BERT_SS_1(args)
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
        total_losses = []
        # latest_memory = np.zeros([1, args.hidden_size], dtype=np.float32)
        for epoch_iter in range(args.epoch):
            latest_memory = np.zeros([1, args.hidden_size], dtype=np.float32)
            for batch_idx, batch_train_data in enumerate(train_data_generator()):
                with tf.GradientTape(persistent=True) as tape:
                    return_dict1 = model(batch_train_data, latest_memory, training=True)
                    return_dict2 = model(batch_train_data, latest_memory, training=True)
                    loss1 = return_dict1["loss"]
                    loss2 = return_dict2["loss"]
                    loss = (loss1 + loss2) / 2
                    losses.append(loss.numpy())
                    # latest_memory = return_dict["latest_memory"].numpy()
                    contra_loss1 = contrastive_loss_v5(return_dict1["hiddens"], return_dict2["hiddens"], \
                                                       batch_train_data.label_ids, batch_train_data.dep_adjs, batch_train_data.attention_mask_ids, \
                                                       jumps=args.gcn_layers, t=args.temperature)
                    contra_loss2 = contrastive_loss_v5(return_dict2["hiddens"], return_dict1["hiddens"], \
                                                       batch_train_data.label_ids, batch_train_data.dep_adjs, batch_train_data.attention_mask_ids, \
                                                       jumps=args.gcn_layers, t=args.temperature)
                    contra_loss = (contra_loss1 + contra_loss2) / 2
                    contra_kl_loss1 = kl_loss(return_dict1["preds"], return_dict2["preds"], batch_train_data.attention_mask_ids)
                    contra_kl_loss2 = kl_loss(return_dict2["preds"], return_dict1["preds"], batch_train_data.attention_mask_ids)
                    contra_kl_loss = (contra_kl_loss1 + contra_kl_loss2) / 2
                    total_loss = loss + args.lambda1 * contra_kl_loss + args.lambda2 * contra_loss
                    total_losses.append(total_loss.numpy())
                grads = tape.gradient(total_loss, tape.watched_variables())
                var_list = tape.watched_variables()
                grads_and_vars = [(grad, var) for grad, var in zip(grads, var_list) if grad != None]
                # optimizer_1.apply_gradients(grads_and_vars=grads_and_vars)
                train_op(grads_and_vars, optimizer_1, optimizer_2)

                step += 1
                if step % args.skep_step == 0:
                    logger.info(f"iteration: {epoch_iter+1} {(step-1)%step_per_epoch+1}/{step_per_epoch}, " \
                                f"step: {step}, loss: {np.mean(losses):>9.6f}, final loss: {np.mean(total_losses):>9.6f}")
                    losses = []
                    total_losses = []
                            
            if args.do_val:
                dev_f1 = evaluate(model, train_reader, dev_data_generator, "dev", logger, latest_memory)                
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    logger.info(f"new best dev f1: {dev_f1}")
                    if args.ckpt is not None and args.ckpt != "None":
                        ckpt_manager.save(checkpoint_number=epoch_iter)
                        logger.info("model saved.")
                                    
            if args.do_test:
                test_f1 = evaluate(model, test_reader, test_data_generator, "test", logger, latest_memory, do_pred=args.do_pred, result_path=os.path.join(output_path, args.results, "test." + str(epoch_iter + 1)))
                if test_f1 > max_test_f1:
                    max_test_f1 = test_f1
                    logger.info(f"new best test f1: {test_f1}")
                                
        logger.info("end training.\n")
    if args.do_val:
        evaluate(model, train_reader, dev_data_generator, "dev", logger, latest_memory)
    if args.do_test:
        evaluate(model, test_reader, test_data_generator, "test", logger, latest_memory, do_pred=args.do_pred, result_path=os.path.join(output_path, args.results, "test.final"))
        
if __name__ == "__main__":
    main()
