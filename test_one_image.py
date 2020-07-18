import tensorflow as tf
import os
import alexnet_model
import resnet_model
import vgg_preprocessing
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BatchSize = 1
_LABEL_CLASSES = 1
resnet_size = 50
data_format = 'channels_first'

def _parse_function_single(image, is_training=False):

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=227,
        output_width=227,
        is_training=is_training
    )
    image = tf.multiply(image, 255, name=None)

    return image

def main():

    params = {
        'resnet_size': resnet_size,
        'data_format': data_format,
        'batch_size': BatchSize,
    }

    features = tf.placeholder(tf.float32, [1, 227, 227, 3])
    features_crop = features[:, 0:224, 0:224, :] / 255.0

    network = alexnet_model.alexnet_memory_generator(keep_prob=0.5,
                                                     data_format='channels_last')
    conv5_1, models = network(inputs=features, is_training=False)

    network2 = resnet_model.imagenet_resnet_v2(
        params['resnet_size'], 71, data_format=params['data_format'])
    models2 = network2(inputs=features_crop, is_training=False)

    concate = tf.concat([models, models2], axis=1)
    results1 = alexnet_model.fc(concate, 6144, 4096, name='fc-euclidean-1')
    results = alexnet_model.fc(results1, 4096, 1, relu=False, name='fc-euclidean-2')

    img_tensor = tf.placeholder(tf.float64, [256, 256, 3])
    img_preprocessed = _parse_function_single(img_tensor)

    with tf.Session() as sess:

        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, "./Models/model.ckpt")

        pic_name = 'example.jpg'
        img = cv2.imread(pic_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img / 255

        img_input = sess.run([img_preprocessed], feed_dict={img_tensor: img})
        my_pred = sess.run([results], feed_dict={features: img_input})
        score = list(my_pred[0])
        score = score[0][0]

        print('The memorability score of {}: {}'.format(pic_name, score))

if __name__ == '__main__':
    main()
