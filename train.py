import subprocess
import time

import tensorflow as tf
from absl import app, flags
from tensorflow.keras.callbacks import (
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.python.client import device_lib

from src import constants, helpers
from src.data.dataset import Dataset
from src.yolo import custom_callbacks
from src.yolo.yolov3 import YOLOv3
from src.visualize import plot_history

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("db", '', "Dataset name.")
flags.DEFINE_string("extra", '', "Extra information to save in the training information file")
# ['none', 'all', 'features', 'last_block', 'last_conv']
trainable = ['none', constants.TRAINABLE_ALL, constants.TRAINABLE_FEATURES, constants.TRAINABLE_LAST_BLOCK,
             constants.TRAINABLE_LAST_CONV]
# Enum
flags.DEFINE_enum("trainable", 'none', trainable,
                  "Use transfer learning from the original weights and keep some part of the network trainable."
                  "none: do not use transfer learning"
                  "all: all the model is trainable"
                  "features: freeze the feature extractor (like DarkNet) and the rest is trainable"
                  "last_block: only the last block of layers is trainable"
                  "last_conv: only the last conv layer is trainable")

# Integers
flags.DEFINE_integer("epochs", 100, "Number of epochs for training")
flags.DEFINE_integer("save_freq", 5, "Checkpoints frequency")
flags.DEFINE_integer("batch_size", 32, "Batch size for the training data")

# Floats
flags.DEFINE_float("lr", 2e-3, "Learning rate")

# Boolean
flags.DEFINE_boolean("tiny", False, "Flag to use tiny version of YOLO")
flags.DEFINE_boolean("use_cosine_lr", True, "Use cosine learning rate scheduler")


def train_detection(_argv):
    gpu_aval = tf.test.is_gpu_available(cuda_only=True)
    gpus = 0
    if gpu_aval:
        for x in device_lib.list_local_devices():
            if x.device_type == "GPU":
                gpus += 1

    print(constants.C_WARNING, "Are CUDA gpus available? \n\t-",
          (constants.C_OKBLUE + ('Yes, ' + str(gpus)) if gpu_aval
           else constants.C_FAIL + 'No.'), constants.C_ENDC)

    batch_size = FLAGS.batch_size
    if gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        # Here the batch size scales up by number of workers since
        # `tf.data.Dataset.batch` expects the global batch size. Previously we used 'batch_size',
        # and now this is multiplied by the number of workers.
        batch_size *= gpus
        with strategy.scope():
            dataset, model = load_model_and_db(batch_size)
    else:
        dataset, model = load_model_and_db(FLAGS.batch_size)

    if FLAGS.db == '':
        tfrecords_pattern_path = constants.PROCESSED_PROJECT_FOLDER_PATH + \
                                 constants.TFRECORDS_PATH.format('COCO', 'val', '*-of-*')
        tensorboard_imgs = custom_callbacks.TensorBoardImagesDetection(
            inference_model=model.inference_model,
            tfrecords_pattern_path=tfrecords_pattern_path,
            dataset_name='COCO',
            model_input_size=model.image_res,
            freq=FLAGS.save_freq,
            logs_path=model.logs_path,
            n_images=5)
        start = time.time()
        history, history_callback = train(model, FLAGS.epochs, dataset.train_data, dataset.validation_data,
                                          FLAGS.save_freq, FLAGS.lr, 'Use fake DS\n', False, True,
                                          extra_callbacks=[tensorboard_imgs])
        helpers.save_history(FLAGS, model.model_name, dataset.dataset_name, history, start, 'detection')
        return model, history

    train_info = FLAGS.extra.replace('\\n', '\n') + '\n'
    train_info += "Train model: {} For {} epochs and {} as Database. \nParameters used:\n    " \
                  "- Checkpoints frequency = {}\n    " \
                  "- Batch size = {}\n".format(model.model_name, FLAGS.epochs, FLAGS.db,
                                               FLAGS.save_freq, FLAGS.batch_size)
    if gpu_aval:
        train_info += "    - {} gpu{} available for training\n".format(gpus, 's' if gpus > 1 else '')
    train_info += "    - Use {} version of the model\n".format('tiny' if FLAGS.tiny else 'full')
    if FLAGS.trainable != 'none':
        train_info += "    - Use transfer learning with trainable option: {} \n".format(FLAGS.trainable)
    else:
        train_info += "    - Train from scratch\n"

    print(constants.C_WARNING, FLAGS.extra.replace('\\n', '\n'), constants.C_ENDC)

    tfrecords_pattern_path = dataset.tf_paths.format(dataset.dataset_name, 'val', '*-of-*')
    tensorboard_imgs = custom_callbacks.TensorBoardImagesDetection(inference_model=model.inference_model,
                                                                   tfrecords_pattern_path=tfrecords_pattern_path,
                                                                   dataset_name=dataset.dataset_name,
                                                                   model_input_size=model.image_res,
                                                                   freq=FLAGS.save_freq,
                                                                   logs_path=model.logs_path,
                                                                   n_images=10)
    start = time.time()
    history, history_callback = train(model=model,
                                      epochs=FLAGS.epochs,
                                      train_data=dataset.train_data,
                                      val_data=dataset.validation_data,
                                      save_freq=FLAGS.save_freq,
                                      initial_lr=FLAGS.lr,
                                      train_info=train_info,
                                      use_fit_generator=False,
                                      use_cosine_lr=FLAGS.use_cosine_lr,
                                      extra_callbacks=[tensorboard_imgs])

    helpers.save_history(FLAGS, model.model_name, dataset.dataset_name, history, start, 'detection')
    return model, history


def load_model_and_db(batch_size):
    model = YOLOv3(tiny=FLAGS.tiny)
    dataset = Dataset(FLAGS.db, FLAGS.tiny)
    dataset.load_datasets(model.image_res, model.anchors, model.masks, batch_size)
    # When loading the model, the folders to save the checkpoints, figures and logs are created.
    if FLAGS.trainable == 'none':
        model.load_models(dataset=dataset,
                          for_training=True,
                          plot_model=False)
    else:
        model.load_for_transfer_learning(dataset, trainable_option=FLAGS.trainable)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    loss = model.get_loss()
    model.train_model.compile(optimizer=optimizer, loss=loss,
                              run_eagerly=False, metrics=['accuracy'])

    return dataset, model


def train(model, epochs: int, train_data, val_data, save_freq: int, initial_lr: float, train_info: str,
          use_fit_generator: bool, use_cosine_lr: bool, save_info: bool = True, show_plots: bool = True,
          extra_callbacks: list = None):
    """
    Standar method to train the model received.
    :param model: Model to train already compiled.
    :param epochs: Number of epochs to train the model.
    :param train_data: Train data as tf.data.Dataset or keras ImageDataGenerator if use_fit_generator.
    :param val_data: Validation data as tf.data.Dataset or keras ImageDataGenerator if use_fit_generator.
    :param save_freq: Checkpoints frequency.
    :param initial_lr: Initial learning rate.
    :param train_info: Training information to save in text fail and print.
    :param use_fit_generator: Flag to use the function fit_generator instead of fit.
    :param use_cosine_lr: Flag to use the cosine decay scheduler.
    :param save_info: Flag to save the training information and the model while training.
    :param show_plots: Flag to show the plots after training, useful to deactivate when running on cluster.
    :param extra_callbacks: List with any extra callbacks to add a part from the default ones in this method.
    :return: history from calling fit(_generator) function, history from custom_callbacks.
    """
    if use_cosine_lr:
        train_info += "    - Use cosine decay scheduler. Initial LR = " + str(initial_lr)
    else:
        train_info += "    - Use constant LR = " + str(initial_lr)
    if save_info:
        with open(model.checkpoints_path + 'train_info.txt', 'w') as t:
            t.write(train_info)
            # last_commit = subprocess.check_output(['git', 'describe', '--always']).strip().decode('UTF-8')
            # t.write('\nLast commit: ' + last_commit + '\n')

    print(constants.C_WARNING, train_info, constants.C_ENDC)

    history_callback = custom_callbacks.History()
    callbacks = [
        EarlyStopping(patience=4, verbose=1),
        history_callback
    ]
    if extra_callbacks is not None:
        for callback in extra_callbacks:
            callbacks.append(callback)

    if save_info:
        callbacks.append(custom_callbacks.ModelCheckpoint(
            model.checkpoints_path + model.model_name + '-epoch-{epoch}-loss-{loss:.5f}.ckpt', monitor='loss',
            verbose=1, save_weights_only=True, save_freq=save_freq, save_best_only=True))
        callbacks.append(custom_callbacks.CustomSaveHistory(model.logs_path + 'train_history.p'))
        callbacks.append(TensorBoard(log_dir=model.logs_path, update_freq='epoch'))

    if use_cosine_lr:
        cosine_lr = custom_callbacks.CosineDecayScheduler(
            initial_lr=initial_lr,
            epochs=epochs,
            epochs_hold_initial_lr=int(epochs / 20), verbose=0)
        callbacks.append(cosine_lr)
    else:
        cosine_lr = None
        callbacks.append(ReduceLROnPlateau(verbose=1, min_lr=1e-5, patience=2))

    if use_fit_generator:
        history = model.train_model.fit_generator(train_data,
                                                  epochs=epochs,
                                                  callbacks=callbacks,
                                                  validation_data=val_data)
    else:
        history = model.train_model.fit(train_data,
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        validation_data=val_data,
                                        use_multiprocessing=True,
                                        workers=8)

    if save_info:
        model.train_model.save_weights(model.checkpoints_path + 'weights.ckpt')

    if use_cosine_lr:
        history.history['learning_rates'] = cosine_lr.learning_rates

    plot_history(history.history, (model.figs_path if save_info else None), show_plots)
    print(constants.C_OKBLUE, "Model finished training. Took in total",
          helpers.display_time(history_callback.total_time, 4),
          constants.C_ENDC)

    for epoch_history in history_callback.history:
        print(epoch_history)

    if save_info:
        with open(model.checkpoints_path + 'train_info.txt', 'a') as t:
            t.write('Model finished training. History:\n')
            for epoch_history in history_callback.history:
                t.write('    ' + epoch_history + '\n')

    return history, history_callback


if __name__ == '__main__':
    app.run(train_detection)
