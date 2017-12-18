

from __future__ import print_function

import os
import zipfile
import subprocess



def get_Imagenet_keys():
    '''
    This maps my tensor names onto the names used by the downloaded ImageNe weights.
    '''

    my_weights = dict()

    #BLOCK
    my_weights['conv_layer_0/conv_kernel_0']= 'conv1_1_W'
    my_weights['conv_layer_0/conv_bias/conv_bias_0']= 'conv1_1_b'

    my_weights['conv_layer_1/conv_kernel_1']= 'conv1_2_W'
    my_weights['conv_layer_1/conv_bias/conv_bias_1']= 'conv1_2_b'

    #BLOCK
    my_weights['conv_layer_3/conv_kernel_3']= 'conv2_1_W'
    my_weights['conv_layer_3/conv_bias/conv_bias_3']= 'conv2_1_b'

    my_weights['conv_layer_4/conv_kernel_4']= 'conv2_2_W'
    my_weights['conv_layer_4/conv_bias/conv_bias_4']= 'conv2_2_b'


    #BLOCK
    my_weights['conv_layer_6/conv_kernel_6']= 'conv3_1_W'
    my_weights['conv_layer_6/conv_bias/conv_bias_6']= 'conv3_1_b'

    my_weights['conv_layer_7/conv_kernel_7']= 'conv3_2_W'
    my_weights['conv_layer_7/conv_bias/conv_bias_7']= 'conv3_2_b'

    my_weights['conv_layer_8/conv_kernel_8']= 'conv3_3_W'
    my_weights['conv_layer_8/conv_bias/conv_bias_8']= 'conv3_3_b'

    #BLOCK
    my_weights['conv_layer_10/conv_kernel_10']= 'conv4_1_W'
    my_weights['conv_layer_10/conv_bias/conv_bias_10']= 'conv4_1_b'

    my_weights['conv_layer_11/conv_kernel_11']= 'conv4_2_W'
    my_weights['conv_layer_11/conv_bias/conv_bias_11']= 'conv4_2_b'

    my_weights['conv_layer_12/conv_kernel_12']= 'conv4_3_W'
    my_weights['conv_layer_12/conv_bias/conv_bias_12']= 'conv4_3_b'

    #BLOCK
    my_weights['conv_layer_14/conv_kernel_14']= 'conv5_1_W'
    my_weights['conv_layer_14/conv_bias/conv_bias_14']= 'conv5_1_b'

    my_weights['conv_layer_15/conv_kernel_15']= 'conv5_2_W'
    my_weights['conv_layer_15/conv_bias/conv_bias_15']= 'conv5_2_b'

    my_weights['conv_layer_16/conv_kernel_16']= 'conv5_3_W'
    my_weights['conv_layer_16/conv_bias/conv_bias_16']= 'conv5_3_b'

    return my_weights

def getCityImages(city, dest_directory = 'imagery/'):
    filename =  city + '.zip'
    if not os.path.isfile(dest_directory+filename):
        cmd = 'gsutil cp gs://sar-store-dl/' + '"' + filename + '" ' + dest_directory
        print('Executing: ') ; print(cmd) ; print(" --- this may take a while.")
        returned = subprocess.call(cmd, shell=True)
        print(returned)

    try:
        zip_ref = zipfile.ZipFile(dest_directory+filename, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()
        print('Extracted image files for: ' + city)

    except:
        print('File not found or unzip error')
