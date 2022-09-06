import random
from collections import namedtuple
import numpy as np
import json
import os


class SemevalAteSSReader(object):
    '''
    SemevalAteSSReader
    '''
    def __init__(self, tokenizer, args):
        self.lower = args.do_lower_case
        self.tokenizer = tokenizer

        self.pad_token = tokenizer.pad_token
        self.pad_id = tokenizer.pad_token_id
        self.cls_token = tokenizer.cls_token
        self.cls_id = tokenizer.cls_token_id
        self.sep_token = tokenizer.sep_token
        self.sep_id = tokenizer.sep_token_id
        self.mask_token = tokenizer.mask_token
        self.mask_id = tokenizer.mask_token_id

        self.current_sample = 0
        self.current_epoch = 0
        self.num_samples = 0

        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.label2id = {'O': 0, 'B-AS': 1, 'I-AS': 2}
        self.id2label = {0: "O", 1: "B-AS", 2: "I-AS"}

    def _read_data(self, input_file, lower=True):
        texts, labels = [], []
        text, label = [], []
        for line in open(input_file, 'r', encoding='utf8'):
            line = line.rstrip()
            # print(list(line))
            if not line:
                if len(text) > 0:
                    texts.append(text)
                    labels.append(label)
                    text, label = [], []
            else:
                word= line.split()
                assert len(word) >= 2, print([word[0]])
                word[0] = word[0].lower() if lower else word[0]
                text.append(word[0])
                label.append(word[-1])
        if len(text) > 0:
            texts.append(text)
            labels.append(label)
        return texts, labels

    def read_labels_file(self, file_path):
        '''
        read dependency / POS / dependency adj labels file
        '''
        with open(file_path, "r", encoding="utf8") as f:
            labels = json.load(f)
    
        return labels

    def _convert_single_data(self, sample):
        tokens = []
        labels = []
        dep_label = []
        pos_label = []
        dep_adj = []
        dep_adj_idx = []
        for i, (word, label, dep, pos, adj) in enumerate(zip(*sample)):
            token = self.tokenizer.tokenize(word)
            tokens += token
            for j,_ in enumerate(token):
                if j == 0:
                    first_label = label
                    labels.append(label)
                    first_dep_label = dep
                    first_pos_label = pos
                    first_dep_adj = np.eye(len(adj))[i].tolist()
                    dep_label.append(dep)
                    pos_label.append(pos)
                    dep_adj.append(adj)
                    # dep_adj_idx.append(i)
                else:
                    if first_label == "O":
                        labels.append("O")
                    else:
                        labels.append("I-AS")
                    dep_label.append(first_dep_label)
                    pos_label.append(first_pos_label)
                    dep_adj.append(first_dep_adj)
                    dep_adj_idx.append(i)
        dep_adj_trans = list(map(list, zip(*dep_adj)))
        new_dep_adj = []
        for idx in range(len(dep_adj_trans)):
            new_dep_adj.append(dep_adj_trans[idx])
            while dep_adj_idx != [] and idx == dep_adj_idx[0]:
                dep_adj_idx.pop(0)
                new_dep_adj.append([0] * len(dep_adj_trans[0]))
        dep_adj = list(map(list, zip(*new_dep_adj)))
        dep_adj = np.sign(np.array(dep_adj) + np.array(new_dep_adj) + np.eye(len(dep_adj)).tolist()).astype(int).tolist()


        # get encoded ids
        sequence = ' '.join(sample[0])
        encoded_sequence = self.tokenizer(sequence)
        token_ids, token_type_ids, attention_mask_ids = encoded_sequence["input_ids"], encoded_sequence["token_type_ids"], encoded_sequence["attention_mask"]
        decoded_sequence = self.tokenizer.decode(token_ids)
        # add label for [CLS] and [SEP]
        tokens = [self.cls_token] + tokens + [self.sep_token]
        labels = ["O"] + labels + ["O"]
        label_ids = [self.label2id[label] for label in labels]
        dep_label = [[0] * len(dep_label[0])] + dep_label + [[0] * len(dep_label[0])]
        pos_label = [[0] * len(pos_label[0])] + pos_label + [[0] * len(pos_label[0])]
        dep_adj = [[0] + dep_adj[i] + [0] for i in range(len(dep_adj))]
        dep_adj = [[0] * len(token_ids)] + dep_adj + [[0] * len(token_ids)]
        assert len(token_ids) == len(labels), f"labels are not matching with the tokens.\n{sequence}"
        # truncate
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + tokens[-1]
            labels = labels[:self.max_seq_len - 1] + labels[-1]
            token_ids = token_ids[:self.max_seq_len - 1] + token_ids[-1]
            token_type_ids = token_type_ids[:self.max_seq_len - 1] + token_type_ids[-1]
            attention_mask_ids = attention_mask_ids[:self.max_seq_len - 1] + attention_mask_ids[-1]
            label_ids = label_ids[:self.max_seq_len - 1] + label_ids[-1]
            dep_label = dep_label[: max_seq_len - 1] + dep_label[-1]
            pos_label = pos_label[: max_seq_len - 1] + pos_label[-1]
            dep_adj = [dep_adj[i][: max_seq_len - 1] + dep_adj[i][-1]  for i in range(max_seq_len - 1)] + [dep_adj[-1][: max_seq_len - 1] + dep_adj[-1][-1]]
        # padding 
        if len(tokens) < self.max_seq_len:
            padding_num = self.max_seq_len - len(tokens)
            tokens = tokens + [self.pad_token] * padding_num
            labels = labels + ["O"] * padding_num
            token_ids = token_ids + [self.pad_id] * padding_num
            token_type_ids = token_type_ids + [self.tokenizer.pad_token_type_id] * padding_num
            attention_mask_ids = attention_mask_ids + [0] * padding_num
            label_ids = label_ids + [self.label2id["O"]] * padding_num
            dep_label = dep_label + [[0] * len(dep_label[0])] * padding_num
            pos_label = pos_label + [[0] * len(pos_label[0])] * padding_num
            dep_adj = [dep_adj[i] + [0] * padding_num for i in range(len(dep_adj))]
            dep_adj = dep_adj + [[0] * len(token_ids)] * padding_num

        output = namedtuple("sample", ["tokens", "labels", "token_ids", "token_type_ids", "attention_mask_ids", "label_ids", "dep_labels", "pos_labels", "dep_adjs"])
        return output(tokens=tokens,
                      labels=labels,
                      token_ids=token_ids,
                      token_type_ids=token_type_ids,
                      attention_mask_ids=attention_mask_ids,
                      label_ids=label_ids,
                      dep_labels=dep_label,
                      pos_labels=pos_label,
                      dep_adjs=dep_adj)
    
    def _convert_data(self, samples, buffer=1024, shuffle=False):
        converted_samples = []
        for sample in samples:
            if len(converted_samples) == buffer:
                if shuffle:
                    random.shuffle(converted_samples)
                for converted_sample in converted_samples:
                    yield converted_sample
                converted_samples = []

            record = self._convert_single_data(sample)
            converted_samples.append(record)

        if len(converted_samples) > 0:
            if shuffle:
                random.shuffle(converted_samples)
            for converted_sample in converted_samples:
                yield converted_sample

    def _prepare_batch_data(self, samples):
        batch_token_ids = np.array([sample.token_ids for sample in samples])
        batch_token_type_ids = np.array([sample.token_type_ids for sample in samples])
        batch_attention_mask_ids = np.array([sample.attention_mask_ids for sample in samples])
        batch_label_ids = np.array([sample.label_ids for sample in samples])
        batch_dep_labels = np.array([sample.dep_labels for sample in samples])
        batch_pos_labels = np.array([sample.pos_labels for sample in samples])
        batch_dep_adjs = np.array([sample.dep_adjs for sample in samples], dtype=np.float32)
        batch_tokens = [sample.tokens for sample in samples]

        output = namedtuple("sample", ["token_ids", "token_type_ids", "attention_mask_ids", "label_ids", "dep_labels", "pos_labels", "dep_adjs", "tokens"])
        return output(token_ids=batch_token_ids,
                      token_type_ids=batch_token_type_ids,
                      attention_mask_ids=batch_attention_mask_ids,
                      label_ids=batch_label_ids,
                      dep_labels=batch_dep_labels,
                      pos_labels=batch_pos_labels,
                      dep_adjs=batch_dep_adjs,
                      tokens=batch_tokens)
    
    def get_step_per_epoch(self):
        return self.step_per_epoch

    def remove_pad(self, return_dict):
        labels, preds, lengths = return_dict["labels"], return_dict["preds"], return_dict["lengths"]
        labels = labels.numpy().tolist()
        preds = preds.numpy().tolist()
        lengths = lengths.numpy().tolist()

        # lengths = [sum(m) for m in mask]
        labels = [label[1:length - 1] for label, length in zip(labels, lengths)]
        preds = [pred[1:length - 1] for pred, length in zip(preds, lengths)]

        y_true, y_pred = [], []
        for label in labels:
            y_true += [self.id2label[l] for l in label]
        for pred in preds:
            y_pred += [self.id2label[np.argmax(p, axis=-1)] for p in pred]
        return y_true, y_pred
    
    def save_predict(self, batch_data, return_dict, filepath):
        labels, preds, lengths = return_dict["labels"], return_dict["preds"], return_dict["lengths"]
        labels = labels.numpy().tolist()
        preds = preds.numpy().tolist()
        lengths = lengths.numpy().tolist()
        tokens = batch_data.tokens

        # lengths = [sum(m) for m in mask]
        labels = [label[1:length - 1] for label, length in zip(labels, lengths)]
        preds = [pred[1:length - 1] for pred, length in zip(preds, lengths)]
        tokens = [token[1:length - 1] for token, length in zip(tokens, lengths)]

        if os.path.exists(filepath):
            f = open(filepath, "a", encoding="utf8")
        else:
            f = open(filepath, "w", encoding="utf8")
            
        y_true, y_pred = [], []
        for token, label, pred in zip(tokens, labels, preds):
            for t, l, p in zip(token, label, pred):
                f.write(' '.join([t, self.id2label[l], self.id2label[np.argmax(p, axis=-1)]]) + '\n')
            f.write('\n')
            
        f.close()

    def data_generator(self, input_file, dep_label_file, pos_label_file, dep_adj_file, batch_size=None, shuffle=False, dev_split=0):
        '''
        data generator
        '''
        if batch_size is None:
            batch_size = self.batch_size
        texts, labels = self._read_data(input_file, lower=self.lower)
        dep_labels = self.read_labels_file(dep_label_file)
        pos_labels = self.read_labels_file(pos_label_file)
        dep_adjs = self.read_labels_file(dep_adj_file)
        samples = list(zip(texts, labels, dep_labels, pos_labels, dep_adjs))
        if shuffle:
            random.shuffle(samples)

        if dev_split != 0:
            if dev_split < 1:
                dev_count = int(len(samples) * dev_split)
            else:
                dev_count = int(dev_split)
            samples, dev_samples = samples[:-dev_count], samples[-dev_count:]
        
        self.step_per_epoch = (len(samples) - 1) // batch_size + 1

        def dev_wrapper():
            batch_data = []
            self.current_sample = 0
            for sample in self._convert_data(dev_samples):
                self.current_sample += 1
                if len(batch_data) < batch_size:
                    batch_data.append(sample)
                if len(batch_data) == batch_size:
                    yield self._prepare_batch_data(batch_data)
                    batch_data = []
            if batch_data:
                yield self._prepare_batch_data(batch_data)

        def wrapper():
            if shuffle:
                random.shuffle(samples)
            batch_data = []
            self.current_sample = 0
            for sample in self._convert_data(samples, shuffle=shuffle):
                self.current_sample += 1
                if len(batch_data) < batch_size:
                    batch_data.append(sample)
                if len(batch_data) == batch_size:
                    yield self._prepare_batch_data(batch_data)
                    batch_data = []
            if batch_data:
                yield self._prepare_batch_data(batch_data)

        if dev_split == 0:
            return wrapper
        else:
            return wrapper, dev_wrapper

