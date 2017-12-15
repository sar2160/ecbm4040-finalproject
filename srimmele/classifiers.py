
from ecbm4040.cnn_funcs import conv_layer, max_pooling_layer, fc_layer
from UrbanCNN.utils import generator_from_file
import tensorflow as tf

import time

class VGGBlockAssembler(object):

    def __init__(self, input_x, n_conv_layers,
                conv_featmap, conv_kernel_size,
                pooling_size, channel_num = 3, seed = 26, indexer = 0):

        assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)
        
        self.n_conv_layers = n_conv_layers
        self.layers = []


        self.layers.append(conv_layer(input_x=input_x,
                                      in_channel=input_x.shape[3],
                                      out_channel=conv_featmap[0],
                                      kernel_shape=conv_kernel_size[0],
                                      rand_seed=seed, index = indexer + len(self.layers)))


        for l in range(n_conv_layers - 1):
                self.layers.append( conv_layer(input_x= self.layers[l].output(),
                                      in_channel=conv_featmap[l],
                                      out_channel=conv_featmap[l+1],
                                      kernel_shape=conv_kernel_size[l+1],
                                      rand_seed=seed, index =  indexer + len(self.layers)))


        self.layers.append(max_pooling_layer(input_x= self.layers[-1].output(),
                                                    k_size=pooling_size[0],
                                                    padding="SAME"))

    def return_index(self):
        return len(self.layers)


def VGG16(input_x, input_y,
    conv_feat_dict, conv_kernel_dict, fc_units,
    pooling_size_dict, channel_num= 3, output_size = 10,
    l2_norm=0.01, seed = 26):

    ## Block 0
    indexer = 0

    Block_0 = VGGBlockAssembler(input_x = input_x, n_conv_layers = 2,
                                conv_featmap = conv_feat_dict[0], conv_kernel_size = conv_kernel_dict[0],
                                pooling_size = pooling_size_dict[0], seed = seed , indexer = indexer)
    indexer += Block_0.return_index()

    ### Block 1

    Block_1 = VGGBlockAssembler(input_x = Block_0.layers[-1].output() , n_conv_layers = 2,
                                conv_featmap = conv_feat_dict[1], conv_kernel_size = conv_kernel_dict[1],
                                pooling_size = pooling_size_dict[1], seed = seed, indexer = indexer)

    indexer += Block_1.return_index()


    ### Block 2

    Block_2 = VGGBlockAssembler(input_x = Block_1.layers[-1].output() , n_conv_layers = 3,
                                conv_featmap = conv_feat_dict[2], conv_kernel_size = conv_kernel_dict[2],
                                pooling_size = pooling_size_dict[2], seed = seed, indexer = indexer)

    indexer += Block_2.return_index()

    ### Block 3

    Block_3 = VGGBlockAssembler(input_x = Block_2.layers[-1].output() , n_conv_layers = 3,
                                conv_featmap = conv_feat_dict[3], conv_kernel_size = conv_kernel_dict[3],
                                pooling_size = pooling_size_dict[3], seed = seed,  indexer = indexer)

    indexer += Block_3.return_index()

    ### Block 4

    Block_4 = VGGBlockAssembler(input_x = Block_3.layers[-1].output() , n_conv_layers = 3,
                                conv_featmap = conv_feat_dict[4], conv_kernel_size = conv_kernel_dict[4],
                                pooling_size = pooling_size_dict[4], seed = seed,  indexer = indexer)

    # flatten
    pool_shape = Block_4.layers[-1].output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(Block_4.layers[-1].output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0, dropout = 0.5)



    fc_layer_1 = fc_layer(input_x= fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1, dropout = 0.5)



    fc_layer_2 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)




    VGG_blocks = [Block_0, Block_1, Block_2, Block_3 , Block_4]
    conv_weights = []
    for block in VGG_blocks:
        conv_weights.append(([b.weight for b in block.layers if isinstance(b, conv_layer)]))
        
       
    conv_weights = [y for x in conv_weights for y in x]
    fc_weights   = [fc_layer_0.weight, fc_layer_1.weight, fc_layer_2.weight]
# loss

    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_weights])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_weights])

        label = tf.one_hot(input_y, 10)

        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits= fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('VGG_16_Loss', loss)

    return fc_layer_2.output(), loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('VGG16_error_num', error_num)
    return error_num



# training function for the VGG Model
def training(train_generator, validation_generator,
             conv_feat_dict,
             fc_units,
             conv_kernel_dict,
             pooling_size_dict,
             img_size,
             l2_norm=0.01,
             seed=26,
             learning_rate=1e-2,
             lr_decay = 2,
             epoch=20,
             batch_size=32,
             samples_per_epoch = 2000,
             verbose=False,
             pre_trained_model=None):
    # print("VGG Net. Parameters: ")
    # print("conv_featmap={}".format(conv_featmap))
    # print("fc_units={}".format(fc_units))
    # print("conv_kernel_size={}".format(conv_kernel_size))
    # print("pooling_size={}".format(pooling_size))
    # print("l2_norm={}".format(l2_norm))
    # print("seed={}".format(seed))
    # print("learning_rate={}".format(learning_rate))



    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, img_size, img_size, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = VGG16(input_x =  xs, input_y =  ys,
                         channel_num=3,
                         output_size=10,
                         conv_feat_dict=conv_feat_dict,
                         fc_units=fc_units,
                         conv_kernel_dict=conv_kernel_dict,
                         pooling_size_dict=pooling_size_dict,
                         l2_norm=l2_norm,
                         seed=seed)




    iters = samples_per_epoch // batch_size

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'VGG16_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            if (epc > 0) and (epc % 10 == 0):
                learning_rate /= lr_decay
                print('New learning rate: ' + str(learning_rate))

            for itr in range(iters):
                iter_total += 1


                #### Sub in image generator here
                batch = next(train_generator)

                training_batch_x = batch[0]
                training_batch_y = batch[1]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})
                if iter_total % 10 == 0:

                    # do validation
                    batch = next(train_generator)

                    X_val = batch[0]
                    y_val = batch[1]

                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            samples_per_epoch,
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        #saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
