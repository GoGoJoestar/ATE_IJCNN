import tensorflow as tf

class learning_rate_linear_warmup_and_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, warm_up_rate_or_step, max_train_step=0, end_lr=0):
        self.init_lr = init_lr
        if 0 <= warm_up_rate_or_step <= 1:
            # assert 0 <= warm_up_rate_or_step <= 1, "warmup rate must between 0 and 1 if max_train_step is 0!"
            self.warm_up_step = int(max_train_step * warm_up_rate_or_step)
        else:
            self.warm_up_step = int(warm_up_rate_or_step)
        self.max_train_step = max_train_step
        self.end_lr = end_lr

    def __call__(self, step):
        if self.warm_up_step > 0 and step <= self.warm_up_step:
            lr = self.init_lr * step / self.warm_up_step
        elif self.max_train_step > 0:
            lr = tf.keras.optimizers.schedules.PolynomialDecay(self.init_lr, self.max_train_step, end_learning_rate=self.end_lr)(step)
        else:
            lr = self.init_lr
        return lr


def train_op(grads_and_vars, optimizer_1, optimizer_2):
    bert_grads_and_vars = [(tf.clip_by_norm(g, 5.0), v) for g, v in grads_and_vars if "tf_bert_model" in v.name]
    later_grads_and_vars = [(tf.clip_by_norm(g, 5.0), v) for g, v in grads_and_vars if "tf_bert_model" not in v.name]
    optimizer_1.apply_gradients(grads_and_vars=bert_grads_and_vars)
    optimizer_2.apply_gradients(grads_and_vars=later_grads_and_vars)