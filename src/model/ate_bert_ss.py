import tensorflow as tf
from transformers import TFBertModel

def get_initializer():
    return tf.keras.initializers.truncated_normal()

class ATE_BERT_SS(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(args.pretrain_model_path)
        self.dense = tf.keras.layers.Dense(units=args.hidden_size, activation=tf.nn.tanh)
        self.syntactic_layer = syntactic_module(args.hidden_size, args.dropout, args.gcn_layers)
        self.semantic_layer = semantic_module(args.hidden_size)
        self.sentence_rnn_layer = sentence_gru(args.hidden_size)
        self.g_o_dense = tf.keras.layers.Dense(args.hidden_size * 4, activation=tf.nn.sigmoid)
        self.project_layer = tf.keras.layers.Dense(units=3)
        self.dropout = tf.keras.layers.Dropout(args.dropout)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs, former_memory, training=False):
        input_ids = inputs.token_ids
        token_type_ids = inputs.token_type_ids
        attention_mask = inputs.attention_mask_ids
        label_ids = inputs.label_ids
        dep_labels = inputs.dep_labels
        pos_labels = inputs.pos_labels
        dep_adjs = inputs.dep_adjs
        former_memory = tf.constant(former_memory, dtype=tf.float32)
        lengths = tf.reduce_sum(attention_mask, axis=-1)

        # bert model
        bert_output_with_pooling = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, training=training)
        bert_output, bert_cls = bert_output_with_pooling["last_hidden_state"], bert_output_with_pooling["pooler_output"]
        bert_output = self.dense(bert_output)
        bert_output = self.dropout(bert_output, training=training)
        bert_cls = bert_output[:,0,:]
        # bert_cls = tf.reduce_sum(bert_output * tf.expand_dims(tf.cast(attention_mask, dtype=tf.float32), axis=-1), axis=-1) / tf.expand_dims(tf.cast(lengths, dtype=tf.float32), axis=-1)

        # syntactic module
        syntactic_outputs = self.syntactic_layer(bert_output, dep_labels, pos_labels, dep_adjs, training=training)
        syntactic_outputs = self.dropout(syntactic_outputs, training=training)

        # semantic module
        semantic_outputs = self.semantic_layer(bert_output, lengths)
        semantic_outputs = self.dropout(semantic_outputs, training=training)

        if training:
            prev_memory, latest_memory = self.sentence_rnn_layer(bert_cls, former_memory)
        else:
            prev_memory = tf.tile(former_memory, [bert_cls.shape[0], 1])
            latest_memory = former_memory

        # g_o = tf.expand_dims(self.g_o_dense(tf.concat([bert_cls, prev_memory], axis=-1)), axis=1)
        # gated_outputs = g_o * tf.concat([syntactic_outputs, semantic_outputs], axis=-1)
        outputs = tf.concat([syntactic_outputs, semantic_outputs], axis=-1)

        logits = self.project_layer(outputs)
        preds = tf.nn.softmax(logits, axis=-1)

        batch_mask = tf.sequence_mask(lengths, maxlen=attention_mask.shape[-1])
        masked_preds = tf.boolean_mask(preds, batch_mask)
        masked_targets = tf.boolean_mask(label_ids, batch_mask)
        loss = self.loss(masked_targets, masked_preds)
        loss = tf.reduce_mean(loss)
        return_vars = {"preds": preds,
                       "labels": label_ids,
                       "mask": attention_mask,
                       "lengths": lengths,
                       "loss": loss,
                       "latest_memory": latest_memory}
        return return_vars

        
class sentence_gru(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.sentence_gru = tf.keras.layers.GRU(hidden_size, return_sequences=True, return_state=True)

    def call(self, inputs, former_memory):
        gru_inputs = tf.expand_dims(inputs, axis=0)
        gru_outputs, latest_memory = self.sentence_gru(gru_inputs, initial_state=former_memory)
        # prev_memory = tf.concat([former_memory, tf.squeeze(gru_outputs, axis=0)[:-1]], axis=0)
        prev_memory = tf.tile(former_memory, [inputs.shape[0], 1])
        return prev_memory, latest_memory


class semantic_module(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
        self.dense = tf.keras.layers.Dense(hidden_size, activation=tf.nn.tanh)

    def call(self, inputs, lengths):
        mask = tf.sequence_mask(lengths, maxlen=inputs.shape[1])
        outputs = self.bilstm(inputs, mask=mask)
        outputs = self.dense(outputs)
        return outputs


class syntactic_module(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate, n_gcn_layers):
        super().__init__()
        self.dep_project = tf.keras.layers.Dense(hidden_size / 2)
        self.pos_project = tf.keras.layers.Dense(hidden_size / 2)
        self.gcn = GCN(hidden_size * 2, n_gcn_layers, dropout_rate)

    def call(self, inputs, dep_labels, pos_labels, dep_adjs, training):
        dep_embedding = self.dep_project(dep_labels)
        pos_embedding = self.pos_project(pos_labels)
        h_syn = tf.concat([dep_embedding, pos_embedding], axis=-1)

        # outputs = h_syn
        outputs = tf.concat([inputs, h_syn], axis=-1)
        outputs = self.gcn(outputs, dep_adjs, training=training)

        return outputs

class GCN(tf.keras.Model):
    def __init__(self, hidden_size, n_layers, dropout_rate):
        super().__init__()
        self.gcn_layers = [tf.keras.layers.Dense(hidden_size) for _ in range(n_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, dep_adjs, training=False):
        # dep_adjs = tf.tile(tf.expand_dims(tf.eye(dep_adjs.shape[-1]), axis=0), [dep_adjs.shape[0], 1, 1])
        n_edges = tf.reduce_sum(dep_adjs, axis=-1, keepdims=True)
        gcn_inputs = inputs
        for i, layer in enumerate(self.gcn_layers):
            inputs_w = layer(gcn_inputs)
            A_inputs_w = tf.math.divide_no_nan(tf.matmul(dep_adjs, inputs_w), n_edges)
            activated = tf.nn.tanh(A_inputs_w)
            gcn_inputs = self.dropout(activated, training=training) if i != len(self.gcn_layers) - 1 else activated
        
        return gcn_inputs

class GAT(tf.keras.Model):
    def __init__(self, hidden_size, n_layers, dropout_rate):
        super().__init__()
        self.gcn_layers = [tf.keras.layers.Dense(hidden_size) for _ in range(n_layers)]
        self.att_weight = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.sqrt_att_size = tf.math.sqrt(tf.cast(hidden_size, dtype=tf.float32))

    def call(self, inputs, dep_adjs, training=False):
        mask_for_att = tf.where(tf.equal(dep_adjs, 1), tf.zeros_like(dep_adjs), tf.ones_like(dep_adjs) * -1000)
        attention_score = tf.matmul(self.att_weight(inputs), inputs, transpose_b=True)
        dk = tf.cast(self.sqrt_att_size, dtype=attention_score.dtype)
        attention_score = tf.math.divide(attention_score, dk)
        attention_score = tf.add(attention_score, tf.cast(mask_for_att, dtype=attention_score.dtype))
        att_adjs = tf.nn.softmax(attention_score, axis=-1)
        
        # n_edges = tf.reduce_sum(dep_adjs, axis=-1, keepdims=True)
        gcn_inputs = inputs
        for i, layer in enumerate(self.gcn_layers):
            inputs_w = layer(gcn_inputs)
            A_inputs_w = tf.matmul(att_adjs, inputs_w)
            activated = tf.nn.tanh(A_inputs_w)
            gcn_inputs = self.dropout(activated, training=training) if i != len(self.gcn_layers) - 1 else activated
        
        return gcn_inputs


class ATE_BERT_SS_1(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(args.pretrain_model_path)
        self.dense = tf.keras.layers.Dense(units=args.hidden_size, activation=tf.nn.tanh)
        self.syntactic_layer = syntactic_module_1(args.hidden_size, args.dropout, args.gcn_layers)
        self.semantic_layer = semantic_module(args.hidden_size)
        self.sentence_rnn_layer = sentence_gru(args.hidden_size)
        # self.g_o_dense = tf.keras.layers.Dense(args.hidden_size * 4, activation=tf.nn.sigmoid)
        self.gated_layer = gated_module(args.hidden_size)
        # self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        # self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        # self.layer_norm_3 = tf.keras.layers.LayerNormalization()
        self.self_att = self_attention(args.hidden_size, args.dropout)
        self.gcn = GAT(args.hidden_size, args.gcn_layers, args.dropout)
        self.project_layer = tf.keras.layers.Dense(units=3)
        self.dropout = tf.keras.layers.Dropout(args.dropout)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, former_memory, training=False):
        input_ids = inputs.token_ids
        token_type_ids = inputs.token_type_ids
        attention_mask = inputs.attention_mask_ids
        label_ids = inputs.label_ids
        dep_labels = inputs.dep_labels
        pos_labels = inputs.pos_labels
        dep_adjs = inputs.dep_adjs
        former_memory = tf.constant(former_memory, dtype=tf.float32)
        lengths = tf.reduce_sum(attention_mask, axis=-1)

        # bert model
        bert_output_with_pooling = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, training=training)
        bert_output, bert_cls = bert_output_with_pooling["last_hidden_state"], bert_output_with_pooling["pooler_output"]
        bert_output = self.dense(bert_output)
        bert_output = self.dropout(bert_output, training=training)
        # bert_cls = bert_output[:,0,:]
        # bert_cls = tf.reduce_sum(bert_output * tf.expand_dims(1. * attention_mask, axis=-1), axis=-1) / tf.expand_dims(1. * lengths, axis=-1)

        # syntactic module
        # dep_embedding, pos_embedding = self.syntactic_layer(dep_labels, pos_labels)
        # dep_embedding = self.dropout(dep_embedding, training=training)
        # pos_embedding = self.dropout(pos_embedding, training=training)

        # semantic module
        # semantic_outputs = self.semantic_layer(bert_output, lengths)
        # semantic_outputs = self.dropout(semantic_outputs, training=training)

        # bert_cls = tf.reduce_sum(semantic_outputs * tf.expand_dims(tf.cast(attention_mask, tf.float32), axis=-1), axis=1) / tf.expand_dims(tf.cast(lengths, tf.float32), axis=-1)

        # dep_embedding = self.layer_norm_1(dep_embedding)
        # pos_embedding = self.layer_norm_2(pos_embedding)
        # semantic_outputs = self.layer_norm_3(semantic_outputs)

        # if training:
        #     prev_memory, latest_memory = self.sentence_rnn_layer(bert_cls, former_memory)
        # else:
        prev_memory = tf.tile(former_memory, [bert_cls.shape[0], 1])
        latest_memory = former_memory
        # prev_memory = self.dropout(prev_memory, training=training)
        # latest_memory = self.dropout(latest_memory, training=training)

        outputs = bert_output
        # outputs = self.gated_layer(bert_output, bert_cls, prev_memory, dep_embedding, pos_embedding)
        # outputs, att_prob, att_score = self.self_att(outputs, attention_mask, training=training)
        # outputs = self.gcn(outputs, dep_adjs, training=training)
        # outputs = self.dropout(outputs, training=training)

        droped_outputs = self.dropout(outputs, training=training)
        logits = self.project_layer(droped_outputs)
        preds = tf.nn.softmax(logits, axis=-1)

        # remove cls
        preds_ = preds[:, 1:, :]
        label_ids_ = label_ids[:, 1:]
        # remove sep & pad
        batch_mask = tf.sequence_mask(lengths - 2, maxlen=attention_mask.shape[-1] - 1)
        # batch_mask = tf.sequence_mask(lengths, maxlen=attention_mask.shape[-1])
        masked_preds = tf.boolean_mask(preds_, batch_mask)
        masked_targets = tf.boolean_mask(label_ids_, batch_mask)

        loss = self.loss(masked_targets, masked_preds)
        loss = tf.reduce_mean(loss)
        # att_loss = attention_loss(dep_adjs, att_prob, attention_mask)
        # loss = loss + 0.2 * att_loss
        # contra_loss = contrastive_loss(att_score, label_ids, dep_adjs, jumps=2)
        # loss = loss + contra_loss
        # cos_score = cosine_distance(latest_memory, prev_memory[0]) + cosine_distance(bert_cls, tf.tile(latest_memory, [bert_cls.shape[0], 1]))
        # loss = loss + cos_score
        # mse_loss = self.mse(bert_cls, tf.tile(latest_memory, [bert_cls.shape[0], 1])) + self.mse(prev_memory, latest_memory)
        # loss = loss + mse_loss
        return_vars = {"preds": preds,
                       "labels": label_ids,
                       "lengths": lengths,
                       "mask": attention_mask,
                       "loss": loss,
                       "latest_memory": latest_memory,
                       "hiddens": outputs}
        return return_vars

class syntactic_module_1(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate, n_gcn_layers):
        super().__init__()
        self.dep_project = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.pos_project = tf.keras.layers.Dense(hidden_size, use_bias=False)

    def call(self, dep_labels, pos_labels):
        n_deps = tf.reduce_sum(dep_labels, axis=-1, keepdims=True)
        dep_embedding = tf.math.divide_no_nan(self.dep_project(dep_labels), tf.cast(n_deps, dtype=tf.float32))
        pos_embedding = self.pos_project(pos_labels)
        
        return dep_embedding, pos_embedding

class gated_module(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        # self.dense_1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid)
        # self.dense_2 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid)
        # self.dense_3 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid)
        self.dense_4 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.tanh)
        self.dense_5 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.tanh)
        self.dense_6 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.tanh)

    def call(self, inputs, cls_inputs, prev_memory, dep_emb, pos_emb):
        # prev_memory = tf.expand_dims(prev_memory, axis=-1)
        # memory = tf.tile(tf.expand_dims(tf.concat([cls_inputs, prev_memory], axis=-1), axis=1), [1, inputs.shape[1], 1])
        # g_dep = self.dense_1(tf.concat([memory, dep_emb], axis=-1))
        # g_pos = self.dense_2(tf.concat([memory, pos_emb], axis=-1))
        # g_sem = self.dense_3(tf.concat([memory, inputs], axis=-1))

        # g_dep = tf.expand_dims(self.dense_1(tf.concat([cls_inputs, prev_memory], axis=-1)), axis=1)
        # g_pos = tf.expand_dims(self.dense_2(tf.concat([cls_inputs, prev_memory], axis=-1)), axis=1)
        # g_sem = tf.expand_dims(self.dense_3(tf.concat([cls_inputs, prev_memory], axis=-1)), axis=1)

        # outputs = g_dep * self.dense_4(dep_emb) + g_pos * self.dense_5(pos_emb) + g_sem * self.dense_6(inputs)
        outputs = self.dense_4(dep_emb) + self.dense_5(pos_emb) + self.dense_6(inputs)
        return outputs

def cosine_distance(a, b):
    pooled_a = tf.math.sqrt(tf.reduce_sum(a * a, axis=-1))
    pooled_b = tf.math.sqrt(tf.reduce_sum(b * b, axis=-1))
    mul_a_b = tf.reduce_sum(a * b, axis=-1)
    cos_score = tf.math.divide_no_nan(mul_a_b, pooled_a * pooled_b)
    return tf.reduce_mean(1 - cos_score)

def cosine_similarity(a, b):
    pooled_a = tf.math.sqrt(tf.reduce_sum(a * a, axis=-1))
    pooled_b = tf.math.sqrt(tf.reduce_sum(b * b, axis=-1))
    mul_a_b = tf.matmul(a, b, transpose_b=True)
    mul_pooled_a_b = tf.matmul(tf.expand_dims(pooled_a, axis=-1), tf.expand_dims(pooled_b, axis=-1), transpose_b=True)
    cos_score = tf.math.divide_no_nan(mul_a_b, mul_pooled_a_b)
    return cos_score

class self_attention(tf.keras.Model):
    def __init__(self, attention_hidden_size, dropout_rate):
        super().__init__()
        self.query_layer = tf.keras.layers.Dense(attention_hidden_size)
        self.key_layer = tf.keras.layers.Dense(attention_hidden_size)
        self.value_layer = tf.keras.layers.Dense(attention_hidden_size)
        self.sqrt_att_size = tf.math.sqrt(tf.cast(attention_hidden_size, dtype=tf.float32))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, mask, training=False):
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)

        pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
        mask_for_att = tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1]))
        mask_for_att = tf.where(tf.equal(mask_for_att, 1), tf.zeros_like(mask_for_att), tf.ones_like(mask_for_att) * -1000)

        attention_score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(self.sqrt_att_size, dtype=attention_score.dtype)
        attention_score = tf.math.divide(attention_score, dk)
        attention_score = tf.add(attention_score, tf.cast(mask_for_att, dtype=attention_score.dtype))
        attention_prob = tf.nn.softmax(attention_score, axis=-1)
        droped_attention_prob = self.dropout(attention_prob, training=training)
        attention_outputs = tf.matmul(droped_attention_prob, value)

        return attention_outputs, attention_prob, attention_score

def focal_loss(y_true, y_pred, alpha=None, beta=2.0):
    label_one_hot = tf.one_hot(y_true, depth=3)
    label_count = tf.reduce_sum(label_one_hot, axis=0)
    if alpha == None:
        alpha = 1. - label_count / tf.reduce_sum(label_count)
        alpha = tf.nn.softmax(alpha, axis=-1)
        alpha = alpha / tf.reduce_max(alpha)
        # reciprocal_label_count = 1. / label_count
        # alpha = reciprocal_label_count / tf.reduce_max(reciprocal_label_count)
    focal_loss = -alpha * tf.math.pow(1. - y_pred, beta) * label_one_hot * tf.math.log(y_pred)
    focal_loss = tf.reduce_mean(focal_loss)

    return focal_loss

def attention_loss(dep_adjs, att_prob, mask):
    # n_edges = tf.reduce_sum(dep_adjs, axis=-1, keepdims=True)
    # norm_dep_adjs = tf.math.divide_no_nan(dep_adjs, n_edges)
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1]))
    masked_att_prob = tf.boolean_mask(att_prob, pad_mask)
    masked_dep_adjs = tf.boolean_mask(dep_adjs, pad_mask)
    loss = tf.keras.losses.binary_crossentropy(masked_dep_adjs, masked_att_prob)

    return loss

class senti_loss(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.project = tf.keras.layers.Dense(1)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, targets):
        logits = self.predict(inputs)
        y_true = tf.math.sign(tf.reduce_sum(targets, axis=-1))
        loss = self.loss(y_true, logits)
        loss = tf.reduce_mean(loss)
        return loss

def contrastive_loss(att_score, label_ids, dep_adjs, jumps=1):
    label_mask = tf.math.sign(label_ids)
    masked_att_score = att_score + tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]]) * -1000.
    masked_att_prob = tf.nn.softmax(masked_att_score, axis=-1)
    masked_att_prob = tf.boolean_mask(masked_att_prob, label_mask)

    jump_dep_adjs = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]])
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    score = tf.reduce_sum(masked_att_prob * masked_jump_dep_adjs, axis=-1)
    log_score = -tf.math.log(score)
    loss = tf.reduce_mean(log_score)
    return loss

def contrastive_loss_v2(hiddens1, hiddens2, label_ids, dep_adjs, mask, jumps=1, t=1.0):
    eye = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]])
    label_mask = tf.math.sign(label_ids)

    score = cosine_similarity(hiddens1, hiddens1) * (1 - eye) + cosine_similarity(hiddens1, hiddens2) * eye
    score = score / t
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1]))
    pad_mask = tf.where(tf.equal(pad_mask, 1), tf.zeros_like(pad_mask), tf.ones_like(pad_mask) * -1000)
    masked_score = score + tf.cast(pad_mask, dtype=score.dtype)
    softmax_prob = tf.nn.softmax(masked_score, axis=-1)
    masked_prob = tf.boolean_mask(softmax_prob, label_mask)

    jump_dep_adjs = eye
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    jump_dep_adjs = tf.sign(tf.nn.dropout(jump_dep_adjs, 0.5) + eye)
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    if masked_prob.shape[0] == 0:
        return 0

    sum_score = tf.reduce_sum(masked_prob * masked_jump_dep_adjs, axis=-1)
    neg_log_score = -tf.math.log(tf.clip_by_value(sum_score, 1e-8, 1.0))
    loss = tf.reduce_mean(neg_log_score)
    return loss

def contrastive_loss_v3(hiddens, label_ids, dep_adjs, mask, jumps=1, t=1.0):
    eye = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]], dtype=tf.int64)
    label_mask = tf.math.sign(label_ids)

    score = cosine_similarity(hiddens, hiddens)
    score = score / t
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1])) * (1 - eye)
    pad_mask = tf.where(tf.equal(pad_mask, 1), tf.zeros_like(pad_mask), tf.ones_like(pad_mask) * -1000)
    masked_score = score + tf.cast(pad_mask, dtype=score.dtype)
    softmax_prob = tf.nn.softmax(masked_score, axis=-1)
    masked_prob = tf.boolean_mask(softmax_prob, label_mask)

    jump_dep_adjs = eye
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    # jump_dep_adjs = tf.sign(tf.nn.dropout(jump_dep_adjs, 0.5) + eye)
    jump_dep_adjs = jump_dep_adjs * (1 - eye)
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    if masked_prob.shape[0] == 0:
        return 0

    sum_score = tf.reduce_sum(masked_prob * tf.cast(masked_jump_dep_adjs, dtype=tf.float32), axis=-1)
    neg_log_score = -tf.math.log(tf.clip_by_value(sum_score, 1e-8, 1.0))
    loss = tf.reduce_mean(neg_log_score)
    return loss

def contrastive_loss_v4(hiddens1, hiddens2, label_ids, dep_adjs, mask, jumps=1, t=1.0):
    eye = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]])
    label_mask = tf.math.sign(label_ids)
    expand_label_mask = tf.expand_dims(label_mask, axis=1)
    
    jump_dep_adjs = eye
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    # jump_dep_adjs = tf.sign(tf.nn.dropout(jump_dep_adjs, 0.5) + eye)
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    score = cosine_similarity(hiddens1, hiddens1) * (1 - eye) + cosine_similarity(hiddens1, hiddens2) * eye
    score = score / t
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.sign(tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1])))
    expand_label_mask = tf.sign(expand_label_mask * pad_mask)
    pad_mask = tf.sign(pad_mask - tf.cast(jump_dep_adjs, dtype=tf.int64))
    pad_mask = tf.sign(pad_mask + expand_label_mask + tf.cast(eye, dtype=tf.int64))
    
    cls_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.constant([0] * mask.shape[0]), mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    cls_mask = tf.sign(cls_mask + tf.transpose(cls_mask, perm=[0, 2, 1]))
    sep_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.reduce_sum(mask, axis=-1) - 1, mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    sep_mask = tf.sign(sep_mask + tf.transpose(sep_mask, perm=[0, 2, 1]))
    pad_mask = tf.sign((1 - tf.sign(cls_mask + sep_mask)) * tf.cast(pad_mask, tf.float32))
    
    pad_mask = tf.where(tf.equal(pad_mask, 1), tf.zeros_like(pad_mask), tf.ones_like(pad_mask) * -1000)
    masked_score = score + tf.cast(pad_mask, dtype=score.dtype)
    softmax_prob = tf.nn.softmax(masked_score, axis=-1)
    contra_prob = softmax_prob * tf.cast(expand_label_mask, tf.float32)
    masked_prob = tf.boolean_mask(contra_prob, label_mask)

    if masked_prob.shape[0] == 0:
        return 0

    sum_score = tf.reduce_sum(masked_prob, axis=-1)
    neg_log_score = -tf.math.log(tf.clip_by_value(sum_score, 1e-8, 1.0))
    # loss = tf.reduce_mean(neg_log_score)
    loss = tf.reduce_sum(neg_log_score)
    return loss

def contrastive_loss_v5(hiddens1, hiddens2, label_ids, dep_adjs, mask, jumps=1, t=1.0):
    def get_single_aspect_mask(label_id):
        src_ids = []
        src_id = []
        last_label = 0
        in_label = False
        for i, label in enumerate(label_id):
            if last_label == 0 and label != 0:
                in_label = True
            if last_label != 0 and label ==0:
                src_ids.append(src_id)
                src_id = []
                in_label = False
            if in_label:
                src_id.append(i)
            last_label = label
        if src_id != []:
            src_ids.append(src_id)
        
        masks = []
        for src_id in src_ids:
            mask_tmp = [0] * label_id.shape[0]
            for idx in src_id:
                mask_tmp[idx] = 1
            masks.append(mask_tmp)
        
        return masks        
            
    eye = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]])
    label_mask = tf.math.sign(label_ids)
    expand_label_mask = tf.expand_dims(label_mask, axis=1)
    
    jump_dep_adjs = eye
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    # jump_dep_adjs = tf.sign(tf.nn.dropout(jump_dep_adjs, 0.5) + eye)
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    score = cosine_similarity(hiddens1, hiddens1) * (1 - eye) + cosine_similarity(hiddens1, hiddens2) * eye
    score = score / t
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.sign(tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1])))
    expand_label_mask = tf.sign(expand_label_mask * pad_mask)
    pad_mask = tf.sign(pad_mask - tf.cast(jump_dep_adjs, dtype=tf.int64))
    pad_mask = tf.sign(pad_mask + expand_label_mask + tf.cast(eye, dtype=tf.int64))
    
    cls_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.constant([0] * mask.shape[0]), mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    cls_mask = tf.sign(cls_mask + tf.transpose(cls_mask, perm=[0, 2, 1]))
    sep_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.reduce_sum(mask, axis=-1) - 1, mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    sep_mask = tf.sign(sep_mask + tf.transpose(sep_mask, perm=[0, 2, 1]))
    pad_mask = tf.sign((1 - tf.sign(cls_mask + sep_mask)) * tf.cast(pad_mask, tf.float32))
    
    pad_mask = tf.where(tf.equal(pad_mask, 1), tf.zeros_like(pad_mask), tf.ones_like(pad_mask) * -1000)
    masked_score = score + tf.cast(pad_mask, dtype=score.dtype)
    softmax_prob = tf.nn.softmax(masked_score, axis=-1)
    
    aspect_masks = [get_single_aspect_mask(label_id) for label_id in label_ids]
    contra_prob = []
    for prob, aspect_mask in zip(softmax_prob, aspect_masks):
        for a_m in aspect_mask:
            contra_prob.append(tf.boolean_mask(prob, a_m) * a_m)
    
    # contra_prob = softmax_prob * tf.cast(expand_label_mask, tf.float32)
    # masked_prob = tf.boolean_mask(contra_prob, label_mask)

    if len(contra_prob) == 0:
        return 0
    contra_prob = tf.concat(contra_prob, axis=0)

    sum_score = tf.reduce_sum(contra_prob, axis=-1)
    neg_log_score = -tf.math.log(tf.clip_by_value(sum_score, 1e-8, 1.0))
    loss = tf.reduce_mean(neg_log_score)
    # loss = tf.reduce_sum(neg_log_score)
    return loss

def contrastive_loss_v5_ablation(hiddens1, hiddens2, label_ids, dep_adjs, mask, jumps=1, t=1.0):
    def get_single_aspect_mask(label_id):
        src_ids = []
        src_id = []
        last_label = 0
        in_label = False
        for i, label in enumerate(label_id):
            if last_label == 0 and label != 0:
                in_label = True
            if last_label != 0 and label ==0:
                src_ids.append(src_id)
                src_id = []
                in_label = False
            if in_label:
                src_id.append(i)
            last_label = label
        if src_id != []:
            src_ids.append(src_id)
        
        masks = []
        for src_id in src_ids:
            mask_tmp = [0] * label_id.shape[0]
            for idx in src_id:
                mask_tmp[idx] = 1
            masks.append(mask_tmp)
        
        return masks        
            
    eye = tf.eye(dep_adjs.shape[-1], batch_shape=[dep_adjs.shape[0]])
    label_mask = tf.math.sign(label_ids)
    expand_label_mask = tf.expand_dims(label_mask, axis=1)
    
    jump_dep_adjs = eye
    for i in range(jumps):
        jump_dep_adjs = tf.sign(tf.matmul(jump_dep_adjs, dep_adjs, transpose_b=True))
    # jump_dep_adjs = tf.sign(tf.nn.dropout(jump_dep_adjs, 0.5) + eye)
    masked_jump_dep_adjs = tf.boolean_mask(jump_dep_adjs, label_mask)

    score = cosine_similarity(hiddens1, hiddens1)
    score = score / t
    pad_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, mask.shape[-1], 1])
    pad_mask = tf.sign(tf.math.multiply(pad_mask, tf.transpose(pad_mask, perm=[0, 2, 1])))
    expand_label_mask = tf.sign(expand_label_mask * pad_mask)
    pad_mask = tf.sign(pad_mask - tf.cast(jump_dep_adjs, dtype=tf.int64))
    pad_mask = tf.sign(pad_mask + expand_label_mask + tf.cast(eye, dtype=tf.int64))
    
    cls_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.constant([0] * mask.shape[0]), mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    cls_mask = tf.sign(cls_mask + tf.transpose(cls_mask, perm=[0, 2, 1]))
    sep_mask = tf.tile(tf.expand_dims(tf.one_hot(tf.reduce_sum(mask, axis=-1) - 1, mask.shape[-1]), axis=-1), [1, 1, mask.shape[-1]])
    sep_mask = tf.sign(sep_mask + tf.transpose(sep_mask, perm=[0, 2, 1]))
    pad_mask = tf.sign((1 - tf.sign(cls_mask + sep_mask)) * tf.cast(pad_mask, tf.float32))
    
    pad_mask = tf.where(tf.equal(pad_mask, 1), tf.zeros_like(pad_mask), tf.ones_like(pad_mask) * -1000)
    masked_score = score + tf.cast(pad_mask, dtype=score.dtype)
    softmax_prob = tf.nn.softmax(masked_score, axis=-1)
    
    aspect_masks = [get_single_aspect_mask(label_id) for label_id in label_ids]
    contra_prob = []
    for prob, aspect_mask in zip(softmax_prob, aspect_masks):
        for a_m in aspect_mask:
            contra_prob.append(tf.boolean_mask(prob, a_m) * a_m)
    
    # contra_prob = softmax_prob * tf.cast(expand_label_mask, tf.float32)
    # masked_prob = tf.boolean_mask(contra_prob, label_mask)

    if len(contra_prob) == 0:
        return 0
    contra_prob = tf.concat(contra_prob, axis=0)

    sum_score = tf.reduce_sum(contra_prob, axis=-1)
    neg_log_score = -tf.math.log(tf.clip_by_value(sum_score, 1e-8, 1.0))
    loss = tf.reduce_mean(neg_log_score)
    # loss = tf.reduce_sum(neg_log_score)
    return loss

def kl_loss(y_true, y_pred, mask):
    loss = tf.keras.losses.kl_divergence(y_true, y_pred)
    loss = loss * mask
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss

