import gin
import tensorflow as tf
import logging
import wandb


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, batch_size, total_epochs):

        lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                                                                    initial_learning_rate=0.001,
                                                                    decay_steps=total_epochs,
                                                                    end_learning_rate=0.0001,
                                                                    power=2  # Power=1.0 indicates linear decay, and power=2.0 indicates squared decay.
                                                                    )
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer = self.optimizer,
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             directory=run_paths["path_ckpts_train"],
                                                             max_to_keep = 1)
        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # from_logits=False: output has already been processed through the sigmoid activation function.


        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        #self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_epochs = total_epochs
        self.log_interval = batch_size
        #self.ckpt_interval = ckpt_interval

        # Define class weight
       # self.class_weights = { 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3, 7: 3, 8: 3, 9: 2, 10: 2, 11: 2}

       # self.class_weight_tensor = tf.constant([self.class_weights[i] for i in range(len(self.class_weights))],
       #                                    dtype=tf.float32)

    @tf.function
    def train_step(self, data, labels):

        #class_weights = tf.gather(self.class_weight_tensor, labels)
        #sample_weights = tf.cast(labels, dtype=tf.float32) * class_weights
        # sample_weights = tf.cast(labels > 0, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def validation_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # class_weights = tf.gather(self.class_weight_tensor, labels)
        # sample_weights = tf.cast(labels, dtype=tf.float32) * class_weights
        # sample_weights = tf.cast(labels > 0, dtype=tf.float32)

        predictions = self.model(data, training=False)
        val_loss = self.loss_object(labels, predictions)

        self.val_loss(val_loss)
        self.val_accuracy(labels, predictions)


    def train(self):
        #print(f"no of batches is {self.log_interval}")
        for idx, (data, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(data, labels)

            if step % (1 * self.log_interval) == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for ds_val, val_labels in self.ds_val:
                    self.validation_step(ds_val, val_labels)

                template = 'epochs: {}, Loss: {}, Accuracy: {}, val_Loss: {}, val_accuracy: {}'
                logging.info(template.format(step/self.log_interval,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # wandb logging
                wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                           'val_acc': self.val_accuracy.result() * 100, 'val_loss': self.val_loss.result(),
                           'step': step})

                # Write summary to tensorboard
                # ...

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            # if step % self.ckpt_interval == 0:
            #     logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
            #     # Save checkpoint
            #     # ...
            #     self.checkpoint_manager.save()

            if step % (self.total_epochs * self.log_interval) == 0:
                logging.info(f'Finished training after {step/self.log_interval} epochs.')
                # Save final checkpoint
                # ...
                self.checkpoint_manager.save()
                return self.val_accuracy.result().numpy()