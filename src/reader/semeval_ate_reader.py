import random
from collections import namedtuple
import numpy as np


class SemevalAteReader(object):
    '''
    SemevalAteReader
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

    def _convert_single_data(self, sample):
        tokens = []
        labels = []
        for word, label in zip(*sample):
            token = self.tokenizer.tokenize(word)
            tokens += token
            for j,_ in enumerate(token):
                if j == 0:
                    first_label = label
                    labels.append(label)
                else:
                    if first_label == "O":
                        labels.append("O")
                    else:
                        labels.append("I-AS")
        # get encoded ids
        sequence = ' '.join(sample[0])
        encoded_sequence = self.tokenizer(sequence)
        token_ids, token_type_ids, attention_mask_ids = encoded_sequence["input_ids"], encoded_sequence["token_type_ids"], encoded_sequence["attention_mask"]
        decoded_sequence = self.tokenizer.decode(token_ids)
        # add label for [CLS] and [SEP]
        tokens = [self.cls_token] + tokens + [self.sep_token]
        labels = ["O"] + labels + ["O"]
        label_ids = [self.label2id[label] for label in labels]
        assert len(token_ids) == len(labels), f"labels are not matching with the tokens.\n{sequence}"
        # truncate
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + tokens[-1]
            labels = labels[:self.max_seq_len - 1] + labels[-1]
            token_ids = token_ids[:self.max_seq_len - 1] + token_ids[-1]
            token_type_ids = token_type_ids[:self.max_seq_len - 1] + token_type_ids[-1]
            attention_mask_ids = attention_mask_ids[:self.max_seq_len - 1] + attention_mask_ids[-1]
            label_ids = label_ids[:self.max_seq_len - 1] + label_ids[-1]
        # padding 
        if len(tokens) < self.max_seq_len:
            padding_num = self.max_seq_len - len(tokens)
            tokens = tokens + [self.pad_token] * padding_num
            labels = labels + ["O"] * padding_num
            token_ids = token_ids + [self.pad_id] * padding_num
            token_type_ids = token_type_ids + [self.tokenizer.pad_token_type_id] * padding_num
            attention_mask_ids = attention_mask_ids + [0] * padding_num
            label_ids = label_ids + [self.label2id["O"]] * padding_num

        output = namedtuple("sample", ["tokens", "labels", "token_ids", "token_type_ids", "attention_mask_ids", "label_ids"])
        return output(tokens=tokens,
                      labels=labels,
                      token_ids=token_ids,
                      token_type_ids=token_type_ids,
                      attention_mask_ids=attention_mask_ids,
                      label_ids=label_ids)
    
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

        output = namedtuple("sample", ["token_ids", "token_type_ids", "attention_mask_ids", "label_ids"])
        return output(token_ids=batch_token_ids,
                      token_type_ids=batch_token_type_ids,
                      attention_mask_ids=batch_attention_mask_ids,
                      label_ids=batch_label_ids)
    
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

    def data_generator(self, input_file, batch_size=None, shuffle=False, dev_split=0, few_count=-1):
        '''
        data generator
        '''
        if batch_size is None:
            batch_size = self.batch_size
        texts, labels = self._read_data(input_file, lower=self.lower)
        samples = list(zip(texts, labels))
        if shuffle:
            random.shuffle(samples)

        if dev_split != 0:
            if dev_split < 1:
                dev_count = int(len(samples) * dev_split)
            else:
                dev_count = int(dev_split)
            samples, dev_samples = samples[:-dev_count], samples[-dev_count:]

        if few_count >= 0:
            samples = samples[:few_count]
        
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

