import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes=13, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = [tf.Variable(0, dtype=tf.int32) for _ in range(num_classes)]
        self.fp = [tf.Variable(0, dtype=tf.int32) for _ in range(num_classes)]
        self.tn = [tf.Variable(0, dtype=tf.int32) for _ in range(num_classes)]
        self.fn = [tf.Variable(0, dtype=tf.int32) for _ in range(num_classes)]


    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)

        # Calculate confusion matrix components
        for i in range(self.num_classes):
            self.tp[i].assign_add(tf.reduce_sum(tf.cast((y_pred == i) & (y_true == i), tf.int32)))
            self.fp[i].assign_add(tf.reduce_sum(tf.cast((y_pred == i) & (y_true != i), tf.int32)))
            self.tn[i].assign_add(tf.reduce_sum(tf.cast((y_pred != i) & (y_true != i), tf.int32)))
            self.fn[i].assign_add(tf.reduce_sum(tf.cast((y_pred != i) & (y_true == i), tf.int32)))

    def result(self):
        # Return confusion matrix components as a dictionary

        return {
            "tp": tf.reduce_sum(self.tp),
            "fp": tf.reduce_sum(self.fp),
            "tn": tf.reduce_sum(self.tn),
            "fn": tf.reduce_sum(self.fn)
        }


