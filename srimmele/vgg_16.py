
from ecbm4040.cnn_funcs_imagenet import conv_layer, max_pooling_layer, fc_layer
from UrbanCNN.utils import generator_from_file
import tensorflow as tf
import numpy as np
import time


def VGG16_Too(input_x, input_y, l2_norm=0, seed = 26, output_size=10):

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=3,
                              out_channel= 64,
                              kernel_shape=3,
                              rand_seed=seed)

    conv_layer_1 = conv_layer(input_x=conv_layer_0.output(),
                                  in_channel=64,
                                  out_channel= 64,
                                  kernel_shape=3,
                                  rand_seed=seed)
                                  rand_seed=seed)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=2,
                                        padding="SAME")



    conv_layer_2 = conv_layer(input_x=pooling_layer_0.output(),
                                  in_channel= 64,
                                  out_channel= 128,
                                  kernel_shape=3,
                                  rand_seed=seed)


    conv_layer_3 = conv_layer(input_x=conv_layer_2.output(),
                                      in_channel= 128,
                                      out_channel= 128,
                                      kernel_shape=3,
                                      rand_seed=seed)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_3.output(),
                                        k_size=2,
                                        padding="SAME")




    conv_layer_4 = conv_layer(input_x=pooling_layer_1.output(),
                                  in_channel= 128,
                                  out_channel= 256,
                                  kernel_shape=3,
                                  rand_seed=seed)


    conv_layer_5 = conv_layer(input_x=conv_layer_4.output(),
                                      in_channel= 256,
                                      out_channel= 256,
                                      kernel_shape=3,
                                      rand_seed=seed)

    conv_layer_6 = conv_layer(input_x=conv_layer_5.output(),
                                      in_channel= 256,
                                      out_channel= 256,
                                      kernel_shape=3,
                                      rand_seed=seed)

    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_6.output(),
                                        k_size=2,
                                        padding="SAME")


    conv_layer_7= conv_layer(input_x=pooling_layer_2.output(),
                                  in_channel= 256,
                                  out_channel= 512,
                                  kernel_shape=3,
                                  rand_seed=seed)


    conv_layer_8 = conv_layer(input_x=conv_layer_7.output(),
                                      in_channel= 512,
                                      out_channel= 512,
                                      kernel_shape=3,
                                      rand_seed=seed)

    conv_layer_9 = conv_layer(input_x=conv_layer_8.output(),
                                      in_channel= 512,
                                      out_channel= 512,
                                      kernel_shape=3,
                                      rand_seed=seed)

    pooling_layer_3 = max_pooling_layer(input_x=conv_layer_9.output(),
                                        k_size=2,
                                        padding="SAME")



    conv_layer_10= conv_layer(input_x=pooling_layer_3.output(),
                                  in_channel= 512,
                                  out_channel= 512,
                                  kernel_shape=3,
                                  rand_seed=seed)


    conv_layer_11 = conv_layer(input_x=conv_layer_10.output(),
                                      in_channel= 512,
                                      out_channel= 512,
                                      kernel_shape=3,
                                      rand_seed=seed)

    conv_layer_12 = conv_layer(input_x=conv_layer_11.output(),
                                      in_channel= 512,
                                      out_channel= 512,
                                      kernel_shape=3,
                                      rand_seed=seed)

    pooling_layer_4 = max_pooling_layer(input_x=conv_layer_12.output(),
                                        k_size=2,
                                        padding="SAME")


    # flatten
    pool_shape = pooling_layer_4.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_4.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=4096,
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=4096,
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)

    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight,conv_layer_1.weight, conv_layer_2.weight, \
                conv_layer_3.weight, conv_layer_4.weight ,conv_layer_5.weight\
                conv_layer_6.weight, conv_layer_7.weight ,conv_layer_8.weight\
                conv_layer_9.weight,conv_layer_10.weight,conv_layer_11.weight\
                conv_layer_12.weight]

    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_1.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_1.output(), loss
