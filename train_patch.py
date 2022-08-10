import tensorflow_datasets as tfds
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers.experimental.preprocessing import (RandomFlip,RandomZoom,RandomRotation)
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from kaggle_datasets import KaggleDatasets

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'kaggle_dataset', 'patchtfds',
    'Tên dataset trên kaggle.')

flags.DEFINE_integer(
    'batch_size', 64,
    'BATCH SIZE')

flags.DEFINE_string(
    'model_name', 'vgg16',
    'chọn model theo tên bao gồm vgg16 và resnet50.')

flags.DEFINE_float(
    'drop_out', 0.5,
    'Hệ số Dropout')

flags.DEFINE_string(
    'save_dir', './model_vgg16.h5',
    'nơi lưu model')

flags.DEFINE_integer(
    'num_class', 3,
    'Số lượng class')

flags.DEFINE_float(
    'lr_step1', 0.001,
    'learning rate step 1')

flags.DEFINE_float(
    'lr_step2', 0.0001,
    'learning rate step 2')

flags.DEFINE_float(
    'lr_step3', 0.00001,
    'learning rate step 3')

flags.DEFINE_integer(
    'epochs_step1', 3,
    'epochs step 1')

flags.DEFINE_integer(
    'epochs_step2', 10,
    'epochs step 2')

flags.DEFINE_integer(
    'epochs_step3', 37,
    'epochs step 3')

def get_dataset(dataset,augment=False):
    AUTO = tf.data.experimental.AUTOTUNE
    data_augmentation = tf.keras.Sequential([
        RandomFlip('horizontal'),
        RandomFlip('vertical'),
        RandomZoom(0.2),
        RandomRotation(0.25),
    ])
    
    def get_img(sample):
        label = sample['label']
        image = sample['image']
        image = tf.image.resize(image,(224,224))
        return image,label
    
    dataset = dataset.map(get_img, num_parallel_calls=AUTO)
    dataset = dataset.batch(FLAGS.batch_size,drop_remainder=True)
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),num_parallel_calls=AUTO) 
    dataset = dataset.prefetch(AUTO)
    return dataset

def create_model(net):
    if net == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16 as NNet
        top_layer_nb = 10
    elif net == 'resnet50':
        from tensorflow.keras.applications.resnet50 import ResNet50 as NNet
        top_layer_nb = 46
    base_model = NNet(weights='imagenet', include_top=False, 
                   input_shape=None,pooling='avg')
    return base_model,top_layer_nb

def compile(model,lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        steps_per_execution=800,
    )

def main(argv):
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
    patchfinal = KaggleDatasets().get_gcs_path(FLAGS.kaggle_dataset)
    (train,test,val), ds_info = tfds.load(
        'patch:1.0.0',
        split=['train', 'test', 'val'],
        with_info=True,
        data_dir=patchfinal,
    )
    ds = get_dataset(train,augment=True)
    ds_val = get_dataset(val)
    ds_test = get_dataset(test)
    
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(FLAGS.save_dir,
                                                      options=save_locally,
                                                      monitor='val_accuracy',#val_recall
                                                      verbose=1,
                                                      save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=10)
    callbacks = [checkpointer,early_stopping_cb]
    
    with tpu_strategy.scope():
        base_model, top_layer_nb = create_model(FLAGS.model_name)
        x = base_model.output
        x = tf.keras.layers.Dropout(FLAGS.drop_out)(x)
        preds = Dense(FLAGS.num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=preds) 

        logging.info('Huấn luyện giai đoạn 1 với lớp cuối cùng')
        for layer in model.layers[:-1]:
            layer.trainable = False
        compile(model,lr=FLAGS.lr_step1)
        
    history = model.fit(ds,
                        epochs=FLAGS.epochs_step1,
                        validation_data=ds_val,
                        callbacks=callbacks,
                       )
    
    logging.info('Huấn luyện giai đoạn 2 mở đóng băng từ layer 0 đến top layer nb: %d', top_layer_nb)        
    with tpu_strategy.scope():    
        for layer in model.layers[top_layer_nb:]:
            layer.trainable = True
        compile(model,lr=FLAGS.lr_step2)

    history_1 = model.fit(ds,
                          epochs=FLAGS.epochs_step1 + FLAGS.epochs_step2,
                          initial_epoch=history.epoch[-1],
                          validation_data=ds_val,
                          callbacks=callbacks,
                         )
    
    logging.info('Mở đóng băng mô hình, huấn luyện giai đoạn 3')
    with tpu_strategy.scope():
        for layer in model.layers[:top_layer_nb]:
            layer.trainable = True
        compile(model,lr=FLAGS.lr_step3)

    history_2 = model.fit(ds,
                          epochs=FLAGS.epochs_step1 + FLAGS.epochs_step2 + FLAGS.epochs_step3,
                          initial_epoch=history_1.epoch[-1],
                          validation_data=ds_val,
                          callbacks=callbacks,
                         )
    with tpu_strategy.scope():
        load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.keras.models.load_model(FLAGS.save_dir, options=load_locally)
    logging.info("Kiểm tra trên tập test")
    model.evaluate(ds_test)
    
if __name__ == '__main__':
  app.run(main)
