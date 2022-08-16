import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app
from absl import flags
from absl import logging

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip,RandomZoom,RandomRotation
from tensorflow.keras.layers import Input,Dense,MaxPool2D,GlobalAveragePooling2D,Conv2D
from tensorflow.keras.models import Model,Sequential

from sklearn.metrics import confusion_matrix
from kaggle_datasets import KaggleDatasets

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'kaggle_dataset', 'fulltfds',
    'Tên dataset trên kaggle.')

flags.DEFINE_integer(
    'batch_size', 8,
    'BATCH SIZE')

flags.DEFINE_string(
    'save_dir', './model_vgg16.h5',
    'nơi lưu model')

flags.DEFINE_string(
    'load_dir', '../input/model-save/repo/vgg16.h5',
    'nơi load model')

flags.DEFINE_integer(
    'num_class', 1,
    'Số lượng class')

flags.DEFINE_float(
    'lr_step1', 0.001,
    'learning rate step 1')

flags.DEFINE_float(
    'lr_step2', 0.0001,
    'learning rate step 2')

flags.DEFINE_integer(
    'epochs_step1', 10,
    'epochs step 1')

flags.DEFINE_integer(
    'epochs_step2', 50,
    'epochs step 2')

flags.DEFINE_integer(
    'image_size', 224,
    'image size')

flags.DEFINE_integer(
    'num_channels1', 512,
    'num_channels của conv2d')

flags.DEFINE_integer(
    'num_channels2', 512,
    'num_channels của conv2d')

def get_dataset(dataset,augment=False,img_size=(FLAGS.image_size, FLAGS.image_size)):
    AUTO = tf.data.experimental.AUTOTUNE
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomFlip('vertical'),
        RandomZoom(0.2),
        RandomRotation(0.25),
    ])
    
    def get_img(sample):
        label = sample['label']
        image = tf.image.resize(sample['image'],img_size)
        return image,label
    
    dataset = dataset.map(get_img, num_parallel_calls=AUTO)
    dataset = dataset.batch(FLAGS.batch_size,drop_remainder=True)
    
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO)
    return dataset

def vgg_block(num_channels, num_convs):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, (3, 3),padding='same', activation='relu'))
    blk.add(MaxPool2D((2, 2), strides=(2, 2)))
    return blk

def add_top_layers(model,
                   image_size=(FLAGS.image_size, FLAGS.image_size),
                   num_channels=[FLAGS.num_channels1,FLAGS.num_channels2],
                   num_convs=[1,1],
                   nb_class=FLAGS.num_class):
    
    def add_vgg_blocks(block):
        for num_channel,num_conv in zip(num_channels, num_convs):
            block = vgg_block(num_channel, num_conv)(block)
        pool = GlobalAveragePooling2D()(block)
        return pool
    
    last_kept_layer = model.layers[-4]
    image_input = Input(shape=(image_size + (3,)))
    
    block = last_kept_layer.output
    model0 = Model(inputs=model.inputs, outputs=block)
    block = model0(image_input)
    block = add_vgg_blocks(block)
    
    dense = Dense(nb_class, activation='sigmoid')(block)
    model_addtop = Model(inputs=image_input, outputs=dense)
    return model_addtop

def compile(model,lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
        steps_per_execution=200,
    )
    
def main(argv):
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.TPUStrategy(tpu)
    
    GCS_DS_PATH = KaggleDatasets().get_gcs_path(FLAGS.kaggle_dataset)
    (train, test,val), ds_info = tfds.load(
        'full_images:1.0.0',
        split=['train', 'test', 'val'],
        with_info=True,
        data_dir=GCS_DS_PATH,
    )
    
    ds = get_dataset(train,augment=True)
    ds_val = get_dataset(val)
    ds_test = get_dataset(test)

    
    #call back
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(FLAGS.save_dir,
                                                      options=save_locally,
                                                      monitor='val_accuracy',
                                                      verbose=1,
                                                      save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=20)
    callbacks = [checkpointer,early_stopping_cb]
    
    logging.info('Huấn luyện giai đoạn 1 với các lớp được thêm vào')
    with tpu_strategy.scope():
        patch_model = tf.keras.models.load_model(FLAGS.load_dir,
                                                 compile=False,
                                                )
        
        image_model =add_top_layers(patch_model)
        for layer in image_model.layers[:2]:
            layer.trainable = False
        compile(image_model,lr=FLAGS.lr_step1)
        
    logging.info(image_model.summary())
    history = image_model.fit(ds,
                              epochs=FLAGS.epochs_step1,
                              validation_data=ds_val,
                              callbacks=callbacks,
                             )
    
    logging.info('Mở đóng băng mô hình, huấn luyện giai đoạn 2')
    with tpu_strategy.scope():
        for layer in image_model.layers[:2]:
            layer.trainable = True
        compile(image_model,lr=FLAGS.lr_step2)
    
    history1 = image_model.fit(ds,
                               epochs=FLAGS.epochs_step1 + FLAGS.epochs_step2,
                               initial_epoch=history.epoch[-1],
                               validation_data=ds_val,
                               callbacks=callbacks,
                              )
    
    with tpu_strategy.scope():
        load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.keras.models.load_model(FLAGS.save_dir, options=load_locally)
    logging.info("Kiểm tra trên tập test")
    image_model.evaluate(ds_test)
 
if __name__ == '__main__':
  app.run(main)
