# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 00:25:08 2019

@author: holy_
"""

import os
import time
#import logging
import tensorflow as tf
#from cnn_image_classifier.image_loading import read_img_sets
from image_loading import read_img_sets
import numpy as np
import cv2



def flat_img_shape(img_size, channels):
    return img_size * img_size * channels


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.05, shape=shape))


def conv2d(layer, weights):
    return tf.nn.conv2d(input=layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(layer):
    return tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def new_conv_layer(layer, num_input_channels, filter_size, num_filters, use_pooling=True):

    weights = weight_variable(shape=[filter_size, filter_size, num_input_channels, num_filters])

    biases = bias_variable(shape=[num_filters])

    layer = conv2d(layer, weights) + biases

    if use_pooling:
        layer = max_pool_2x2(layer)

    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

    return layer, num_features


def new_fully_connected_layer(layer, num_inputs, num_outputs, use_relu=True, layer_id=1):

    weights = weight_variable(shape=[num_inputs, num_outputs])

    biases = bias_variable(shape=[num_outputs])

    layer = tf.matmul(layer, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    #if summaries: 
        #tf.summary.histogram("Weight_fc" + str(layer_id), weights)
        #tf.summary.histogram("bias_fc" + str(layer_id), biases)

    return layer

"""
def log_progress(session, saver, cost, accuracy, epoch, test_feed_dict, checkpoint_path):

    val_loss = session.run(cost, feed_dict=test_feed_dict)
    acc = session.run(accuracy, feed_dict=test_feed_dict)

    msg = "Epoch {0} --- Accuracy: {1:>6.1%}, Validation Loss: {2:.3f}"
    logging.info(msg.format(epoch, acc, val_loss))

    save_path = saver.save(session, checkpoint_path)
    logging.debug("Creating resource: CNN Model [%s]", save_path)
"""

def variables(flat_img_size, num_classes):

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, flat_img_size], name='x-input')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        #keep_prob = tf.placeholder(tf.float32)

    return x, y_true#, keep_prob


def model(x, img_size, colour_channels, filter_size, neurons, num_classes):

    with tf.name_scope('reshaping'):
        x_image = tf.reshape(x, [-1, img_size, img_size, colour_channels])
        #tf.summary.image('example_images', x_image)

    with tf.name_scope('Conv1'):
        layer_conv1 = new_conv_layer(
            x_image,
            num_input_channels=colour_channels,
            filter_size=filter_size,
            num_filters=img_size
        )

    with tf.name_scope('Conv2'):
        layer_conv2 = new_conv_layer(
            layer_conv1,
            num_input_channels=img_size,
            filter_size=filter_size,
            num_filters=neurons
        )

    with tf.name_scope('Conv3'):
        layer_conv3 = new_conv_layer(
            layer_conv2,
            num_input_channels=neurons,
            filter_size=filter_size,
            num_filters=neurons
        )

    with tf.name_scope('Fully_Connected1'):

        flat_layer, num_features = flatten_layer(layer_conv3)

        layer_fc1 = new_fully_connected_layer(
            flat_layer,
            num_features,
            num_outputs=1024,
            layer_id=1,
            #summaries=True
        )

#    with tf.name_scope('Dropout'):

#       dropout_layer = dropout(layer_fc1, keep_prob)

    with tf.name_scope('Fully_Connected2'):

        layer_fc2 = new_fully_connected_layer(
            layer_fc1,
            num_inputs=1024,
            num_outputs=num_classes,
            use_relu=False,
            layer_id=2,
            #summaries=True
        )

    return layer_fc2


def softmax(logits):

    with tf.name_scope('softmax'):
        y_pred = tf.nn.softmax(logits, name='output')

    return y_pred



"""
def optimizer(cost):

    with tf.name_scope('train'):
        training_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    return training_op
"""

"""

def restore_or_initialize(session, saver, checkpoint_dir):

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    

    if ckpt:
        logging.debug("Loading resource: CNN Model [%s]", os.path.join(checkpoint_dir, 'model.ckpt'))
        saver.restore(session, ckpt.model_checkpoint_path)

    else:
        logging.warning(
            "Resource not found: CNN Model [%s]. Model will now be trained from scratch.",
            os.path.join(checkpoint_dir, 'model.ckpt'))

        os.makedirs(checkpoint_dir)
        tf.global_variables_initializer().run()
"""

def predict(img_dir, model_dir, img_size=64, colour_channels=3, batch_size=803):

    checkpoint_dir = os.path.join(os.path.abspath(model_dir), 'tensorflow/cnn/model/')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    checkpoint_dir2 = os.path.join(os.path.abspath(model_dir), 'tensorflow/cnn/model2')
    ckpt2 = tf.train.get_checkpoint_state(checkpoint_dir2)
    log_dir = os.path.join(os.path.abspath(model_dir), 'tensorflow/cnn/logs/cnn_with_summaries')
    

    data, category_ref = read_img_sets(img_dir + '/predict', img_size)

    flat_img_size = flat_img_shape(img_size, colour_channels)

    num_classes = len(category_ref)

    x, y_true= variables(flat_img_size, num_classes)
    logits = model(x, img_size, colour_channels, filter_size=3, neurons=2*img_size, num_classes=num_classes)
    predict_op = softmax(logits)

    with tf.Session() as sess:
        
        saver = tf.train.Saver()
        #restore_or_initialize(sess, saver, checkpoint_dir)
        
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph()) 
        writer.add_graph(tf.get_default_graph()) 
        saver.restore(sess, ckpt.model_checkpoint_path) 
        saver.save(sess, checkpoint_dir2)  

        x_predict_batch, y_predict_batch, idss, cls_predict_batch = data.train.next_batch(batch_size=803)
        
        x_predict_batch = x_predict_batch.reshape(batch_size, flat_img_size)
        print(idss[0],idss[1])
        print(x_predict_batch)
        testimg = cv2.imread('test.png')
        testimg = cv2.resize(testimg, (64,64), cv2.INTER_LINEAR)
        #img=img.astype(np.float32)
        testbin = np.empty((1,12288),np.float32)
        #testbin=np.vstack((testbin,testimg.flatten()))
        testbin[0]=testimg.flatten()
        testbin=np.multiply(testbin, 1.0/255.0)
        print(testbin)
        #print(testimg.flatten().shape)
        start = time.time()
        prediction = sess.run([tf.argmax(predict_op, dimension=1)], feed_dict={x: x_predict_batch})
        #prediction = sess.run([tf.argmax(predict_op, dimension=1)], feed_dict={x: testbin})
        
        #print(prediction[0],cls_predict_batch)
		
	  #if category_ref[prediction[0][0]] == cls_predict_batch[0]:
        #    print("it is correct")
	
       
        end = time.time()
        timepershot=(end-start)/batch_size
        
        
        idsspath=img_dir + '/predict/'+cls_predict_batch[0]+'/'+idss[0]
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topleftCornerOfText = (10,40)
        topleftCornerOfText2 = (10,70)
        topleftCornerOfText3 = (10,100)
        bottomleft = (10,350)
        fontScale              = 1
        fontColor              = (0,0,0)
        lineType               = 2
        
        img2 = cv2.imread(idsspath, cv2.IMREAD_COLOR) 
        cv2.putText(img2,'Prediction: %s' % (category_ref[prediction[0][0]]), topleftCornerOfText, font, fontScale,fontColor,lineType)
        cv2.putText(img2,'Truth: %s' % (cls_predict_batch[0]), topleftCornerOfText2, font, fontScale,fontColor,lineType)
        cv2.putText(img2,'Time: %s' % (timepershot), topleftCornerOfText3, font, fontScale,fontColor,lineType)
        cv2.putText(img2,'Path: %s' % (idss[0]),bottomleft, font, fontScale,fontColor,lineType)
        cv2.imshow('imgtest',img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(img_dir + '/predict/'+cls_predict_batch[0]+'/'+idss[0])
        j=0
        for i in range(batch_size):
            if (category_ref[prediction[0][i]] == cls_predict_batch[i]):
               j+=1
            
        print(j,batch_size)    
        return category_ref[prediction[0][0]], cls_predict_batch[0]












