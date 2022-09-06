import tensorflow as tf
from transformers import TFBertModel


class ATE_BERT(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(args.pretrain_model_path)
        self.dense = tf.keras.layers.Dense(units=200, activation=tf.nn.tanh)
        self.project_layer = tf.keras.layers.Dense(units=3)
        self.dropout = tf.keras.layers.Dropout(args.dropout)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs, training=False):
        input_ids = inputs.token_ids
        token_type_ids = inputs.token_type_ids
        attention_mask = inputs.attention_mask_ids
        label_ids = inputs.label_ids
        lengths = tf.reduce_sum(attention_mask, axis=-1)

        bert_output_with_pooling = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, training=training)
        bert_output, bert_cls = bert_output_with_pooling["last_hidden_state"], bert_output_with_pooling["pooler_output"]
        bert_output = self.dropout(bert_output, training=training)
        bert_output = self.dense(bert_output)
        logits = self.project_layer(bert_output)
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
                       "loss": loss}
        return return_vars

        