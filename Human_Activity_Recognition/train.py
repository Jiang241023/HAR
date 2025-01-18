import gin
import tensorflow as tf
import logging
import wandb


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, batch_size, total_epochs,use_polyloss=False, poly_loss_alpha=2, use_rdrop=False, rdrop_alpha=0.1 ,labeling_mode= 'S2L'):

        self.use_polyloss = use_polyloss
        self.poly_loss_alpha = poly_loss_alpha
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha
        self.labeling_mode = labeling_mode

        lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                                                                    initial_learning_rate=0.001,
                                                                    decay_steps=total_epochs,
                                                                    end_learning_rate=0.0004,
                                                                    power=2  # Power=1.0 indicates linear decay, and power=2.0 indicates squared decay.
                                                                    )
        # lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=0.0004,  # Small enough to stabilize training
        #     decay_steps=total_epochs * steps_per_epoch,  # Total number of steps (epochs × batches per epoch)
        #     alpha=0.05  # Final LR will be 5% of initial LR
        # )

        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.checkpoint = tf.train.Checkpoint(optimizer = self.optimizer,
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             directory=run_paths["path_ckpts_train"],
                                                             max_to_keep = 1)
        # Loss selection
        if self.use_polyloss:
            def poly_loss(y_true, y_pred):
                cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
                prob = tf.reduce_max(y_pred, axis=-1)
                poly_term = self.poly_loss_alpha * (1 - prob) ** 2
                return cross_entropy_loss + poly_term

            self.loss_object = poly_loss
        else:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

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

    def compute_rdrop_loss(self, logits1, logits2):
        """
        Compute the R-Drop loss as KL divergence between two logits.
        """
        kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        kl_div = kl_loss(logits1, logits2) + kl_loss(logits2, logits1)
        return tf.reduce_mean(kl_div)

    @tf.function
    def train_step(self, data, labels):
        # Reshape labels to match the output shape of the model
        if self.labeling_mode == 'S2S':
            labels = tf.reshape(labels, (tf.shape(data)[0], tf.shape(data)[1]))  # Shape: (batch_size, time_steps)

        #class_weights = tf.gather(self.class_weight_tensor, labels)
        #sample_weights = tf.cast(labels, dtype=tf.float32) * class_weights
        # sample_weights = tf.cast(labels > 0, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # First forward pass
            logits1 = self.model(data, training=True)
            # Second forward pass (for R-Drop consistency)
            logits2 = self.model(data, training=True) if self.use_rdrop else logits1

            # Calculate the primary loss (e.g., cross-entropy or PolyLoss)
            primary_loss = self.loss_object(labels, logits1)
            total_loss = primary_loss

            # Add R-Drop regularization loss if enabled
            if self.use_rdrop:
                rdrop_loss = self.compute_rdrop_loss(logits1, logits2)
                total_loss += self.rdrop_alpha * rdrop_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(total_loss)
        self.train_accuracy(labels, logits1)

    @tf.function
    def validation_step(self, data, labels):
        # Reshape labels to match the output shape of the model
        if self.labeling_mode == 'S2S':
            labels = tf.reshape(labels, (tf.shape(data)[0], tf.shape(data)[1]))  # Shape: (batch_size, time_steps)
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