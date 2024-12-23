import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import lstm_like, gru_like, transformer_like
import tensorflow as tf
import random
import numpy as np
import os

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
random_seed(47)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True,'Specify whether to train or evaluate a model.')

@gin.configurable
def train_model(model, ds_train, ds_val, batch_size, run_paths, path_model_id):
    print('-' * 88)
    print(f'Starting training {path_model_id}')
    model.summary()
    trainer = Trainer(model, ds_train, ds_val, run_paths, batch_size)
    for layer in model.layers:
        print(layer.name, layer.trainable)
    for _ in trainer.train():
        continue
    # for layer in base_model.layers[-unfrz_layer:]:
    #     layer.trainable = True
    # for _ in trainer.train():
    #     continue
    print(f"Training checkpoint path for {path_model_id}: {run_paths['path_ckpts_train']}")
    print(f'Training completed for {path_model_id}')
    print('-' * 88)

def main(argv):

    # generate folder structures
    # run_paths_1 = utils_params.gen_run_folder(path_model_id = 'lstm_like')
    # run_paths_2 = utils_params.gen_run_folder(path_model_id = 'gru_like')
    run_paths_3 = utils_params.gen_run_folder(path_model_id = 'transformer_like')

    # set loggers
    # utils_misc.set_loggers(run_paths_1['path_logs_train'], logging.INFO)
    # utils_misc.set_loggers(run_paths_2['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_3['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings([r'E:\DL_LAB_HAPT\HAR\Human_Activity_Recognition\configs\config.gin'], [])
    # utils_params.save_config(run_paths_1['path_gin'], gin.config_str())
    # utils_params.save_config(run_paths_2['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_3['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, batch_size = load()

    for batch_data, batch_labels in ds_train.take(1):
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")

    # for name, dataset in datasets:
    #     print(f"Processing the dataset of {name}...")
    #     for window_data, window_labels in dataset.take(1):
    #         print("Window Data Shape: ", window_data.shape)
    #         print("Window Labels Shape :", window_labels.shape)
    #         print("Window Labels : ", window_labels.numpy())
    #         print("=" * 50)
    # model
    # model_1 = lstm_like(input_shape=(128, 6), n_classes=13)

    model_2 = gru_like(input_shape=(128, 6), n_classes=13)

    model_3= transformer_like(input_shape=(128, 6), n_classes=13)


    if FLAGS.train:

        # # Model_1
        # wandb.init(project='Human_Activity_Recognition', name=run_paths_1['model_id'],
        #             config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))# setup wandb
        #
        # train_model(model = model_1,
        #             ds_train = ds_train,
        #             ds_val = ds_val,
        #             batch_size = batch_size,
        #             run_paths = run_paths_1,
        #             path_model_id = 'lstm_like')
        #
        # wandb.finish()

        # # Model_2
        # wandb.init(project='diabetic-retinopathy-detection', name=run_paths_2['model_id'],
        #            config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
        # train_model(model = model_2,
        #             ds_train = ds_train,
        #             ds_val = ds_val,
        #             batch_size = batch_size,
        #             run_paths = run_paths_2,
        #             path_model_id = 'gru_like')
        # wandb.finish()


        # Model_3
        wandb.init(project='diabetic-retinopathy-detection', name=run_paths_3['model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
        train_model(model = model_3,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    batch_size=batch_size,
                    run_paths = run_paths_3,
                    path_model_id = 'transformer_like')
        wandb.finish()

        wandb.init(project='Human_Activity_Recognition', name='evaluation_phase',
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        evaluate(model_1=model_3, model_2= None, model_3=None, ds_test=ds_test, ensemble=False)
        wandb.finish()

    else:
        checkpoint_path_1 = r'E:\DL_LAB_HAPT\dl-lab-24w-team04-har\experiments\run_2024-12-21T15-57-07-823708_lstm_like\ckpts'
        checkpoint_1 = tf.train.Checkpoint(model=model_1)
        latest_checkpoint_1 = tf.train.latest_checkpoint(checkpoint_path_1)
        if latest_checkpoint_1:
            print(f"Restoring from checkpoint_1: {latest_checkpoint_1}")
            checkpoint_1.restore(latest_checkpoint_1)
        else:
            print("No checkpoint found. Starting from scratch.")

        wandb.init(project='Human_Activity_Recognition', name='evaluation_phase',
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        evaluate(model_1=None, model_2=model_1, model_3=None, ds_test=ds_test, ensemble=False)
        # checkpoint_path_1 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-371592_mobilenet_like/ckpts'
        # checkpoint_path_2 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-371988_vgg_like/ckpts'
        # checkpoint_path_3 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-372289_inception_v2_like/ckpts'
        #
        # checkpoint_1 = tf.train.Checkpoint(model = model_1)
        # latest_checkpoint_1 = tf.train.latest_checkpoint(checkpoint_path_1)
        # if latest_checkpoint_1:
        #     print(f"Restoring from checkpoint_1: {latest_checkpoint_1}")
        #     checkpoint_1.restore(latest_checkpoint_1)
        # else:
        #     print("No checkpoint found. Starting from scratch.")
        #
        # # Model_2
        # checkpoint_2 = tf.train.Checkpoint(model = model_2)
        # latest_checkpoint_2 = tf.train.latest_checkpoint(checkpoint_path_2)
        # if latest_checkpoint_2:
        #     print(f"Restoring from checkpoint_2: {latest_checkpoint_2}")
        #     checkpoint_2.restore(latest_checkpoint_2)
        # else:
        #     print("No checkpoint found. Starting from scratch.")
        #
        # # Model_3
        # checkpoint_3 = tf.train.Checkpoint(model = model_3)
        # latest_checkpoint_3 = tf.train.latest_checkpoint(checkpoint_path_3)
        # if latest_checkpoint_3:
        #     print(f"Restoring from checkpoint_3: {latest_checkpoint_3}")
        #     checkpoint_3.restore(latest_checkpoint_3)
        # else:
        #     print("No checkpoint found. Starting from scratch.")


if __name__ == "__main__":
    wandb.login(key="40c93726af78ad0b90c6fe3174c18599ecf9f619")
    app.run(main)