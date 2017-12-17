
from ecbm4040.cnn_funcs import conv_layer, max_pooling_layer, fc_layer
from UrbanCNN.utils import generator_from_file
from srimmele.vgg_16 import VGG16_Too
import tensorflow as tf

import time




def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate):

    with tf.name_scope('train_step'):
        step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred      = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('VGG16_error_num', error_num)
    return error_num

def accuracy(output, input_y):
    with tf.name_scope('evaluate'):
        pred      = tf.argmax(output, axis=1)
        accuracy  = tf.metrics.accuracy(input_y, pred, name = 'accuracy')
        tf.summary.scalar('VGG16_val_accuracy', accuracy)
    return accuracy

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

    output, loss = VGG16_Too(xs, ys, \
                    l2_norm=0, seed = 26, output_size=10)




    iters = samples_per_epoch // batch_size

    step = train_step(loss, learning_rate = learning_rate)
    eve  = evaluate(output, ys)
    #acc  = accuracy(output,ys)

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
            except Exception as e:
                print("Load model Failed!")
                print(e);
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
                    batch = next(validation_generator)

                    X_val = batch[0]
                    y_val = batch[1]

                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc =  100 - ((valid_eve * 100) / y_val.shape[0])

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
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
