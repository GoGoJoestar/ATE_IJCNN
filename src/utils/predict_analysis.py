from collections import namedtuple
import os

def read_res_file(path):
    label2id = {'O': 0, 'B-AS': 1, 'I-AS': 2}
    Sample = namedtuple("Sample", ["tokens", "labels", "preds", "label_ids", "pred_ids"])
    samples = []
    tokens, labels, preds, label_ids, pred_ids = [], [], [], [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip()
            if line == "":
                samples.append(Sample(tokens, labels, preds, label_ids, pred_ids))
                tokens, labels, preds, label_ids, pred_ids = [], [], [], [], []
                continue
            token, label, pred = line.split(' ')
            tokens.append(token)
            labels.append(label)
            preds.append(pred)
            label_ids.append(label2id[label])
            pred_ids.append(label2id[pred])
        if tokens and labels and preds and label_ids and pred_ids:
            samples.append(Sample(tokens, labels, preds, label_ids, pred_ids))
    return samples

def get_single_aspect_mask(label_id):
    src_ids = []
    src_id = []
    last_label = 0
    in_label = False
    count = 0
    for i, label in enumerate(label_id):
        if label == 1:
            count += 1
        if last_label == 0 and label != 0:
            in_label = True
        if in_label and label != 2:
            if src_id != []:
                src_ids.append(src_id)
                src_id = []
            if label == 0:
                in_label = False
        if in_label:
            src_id.append(i)
        last_label = label
    if src_id != []:
        src_ids.append(src_id)
    assert count <= len(src_ids)
    return src_ids

def count_pred(samples):
    pred_count = 0
    gold_count = 0
    overlap_count = 0
    outside_count = 0
    true_pos_count = 0
    for sample in samples:
        golds = get_single_aspect_mask(sample.label_ids)
        preds = get_single_aspect_mask(sample.pred_ids)
        gold_count += len(golds)
        pred_count += len(preds)
        for pred in preds:
            if pred in golds:
                # true positive
                true_pos_count += 1
                continue
            else:
                if golds == []:
                    # predict outside
                    outside_count += 1
                    continue
                for gold in golds:
                    if (set(pred) & set(gold)) != set():
                        # overlap
                        overlap_count += 1
                        outside_count -= 1
                        break
                # predict outside
                outside_count += 1
                
                
    return gold_count, pred_count, true_pos_count, overlap_count, outside_count

def process_file_or_dir(path):
    assert os.path.exists(path) == True
    if os.path.isfile(path):
        counts = count_pred(read_res_file(path))
        return {path: counts}
    else:
        file_paths = os.listdir(path)
        counts = {os.path.join(path, file_path): count_pred(read_res_file(os.path.join(path, file_path))) for file_path in file_paths}
        return counts
        

if __name__ == "__main__":
    # l14
    # res1 = process_file_or_dir("output/2021-12-21_12:05:17/results")#14
    # res2 = process_file_or_dir("output/2021-12-21_09:32:42/results")#6
    res1_5 = process_file_or_dir("output/2021-12-22_10:28:31/results")#13
    # r14
    # res3 = process_file_or_dir("output/2021-12-21_12:05:19/results")#19
    # res4 = process_file_or_dir("output/2021-12-21_09:35:42/results")#30
    res3_5 = process_file_or_dir("output/2021-12-22_10:28:33/results")#13
    # r15
    # res5 = process_file_or_dir("output/2021-12-21_12:05:18/results")#17
    # res6 = process_file_or_dir("output/2021-12-21_09:35:47/results")#23
    res5_5 = process_file_or_dir("output/2021-12-22_10:28:35/results")#30
    # r16
    # res7 = process_file_or_dir("output/2021-12-21_12:05:20/results")#19
    # res8 = process_file_or_dir("output/2021-12-21_09:35:53/results")#26
    res7_5 = process_file_or_dir("output/2021-12-22_10:28:32/results")#21
    x=1
    