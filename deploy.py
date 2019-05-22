# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 00:44:55 2019

@author: holy_
"""

from mvnc import mvncapi as mvnc
import numpy as np
import argparse
import time
import cv2
import os
import sys

counter=0
counter2=0
IMAGE_PATH = 'images/'
checktime=0

def do_initialize(): # -> (mvnc.Device, mvnc.Graph):
    """Creates and opens the Neural Compute device and c
    reates a graph that can execute inferences on it.
    Returns
    -------
    device : mvnc.Device
        The opened device.  Will be None if couldn't open Device.
    graph : mvnc.Graph
        The allocated graph to use for inferences.  Will be None if couldn't allocate graph
    """

    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
            print('Error - No devices found')
            return (None, None)
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    filefolder = os.path.dirname(os.path.realpath(__file__))
    
    ##Change this accordingly
    ########################################################################################
    graph_filename = filefolder + '/modelpredict.graph' 
    ########################################################################################
    try :
        with open(graph_filename, mode='rb') as f:
            in_memory_graph = f.read()
    except :
        print ("Error reading graph file: " + graph_filename)

    graph = device.AllocateGraph(in_memory_graph)

    return device, graph

def do_inference(graph, image_filename):
    labels=['Ben', 'Mal']
    global checktime
    
    image_for_inference = cv2.imread(image_filename)
    image_for_inference = cv2.resize(image_for_inference, (64, 64), cv2.INTER_LINEAR)
    imgbin = np.empty((1,12288),np.float16)
    imgbin[0] = image_for_inference.flatten()
    imgbin = np.multiply(imgbin,1.0/255.0)

    start=time.time()
    graph.LoadTensor(imgbin, None)
    output, userobj = graph.GetResult()
    end=time.time()
    
    checktime=end-start
    return labels, output
    
def do_cleanup(device, graph):
    """Cleans up the NCAPI resources.
    Parameters
    ----------
    device : mvncapi.Device
             Device instance that was initialized in the do_initialize method
    graph : mvncapi.Graph
            Graph instance that was initialized in the do_initialize method
    Returns
    -------
    None
    """
    graph.DeallocateGraph()
    device.CloseDevice()


def show_inference_results(image_filename, infer_labels, infer_probabilities):

    global counter2
    global counter
    global checktime
    
   
    print("\n")
    print('-----------------------------------------------------------')
    correct =(infer_probabilities.argsort()[1])
    
    pathname = os.path.basename(image_filename)
    print(infer_probabilities.argsort()[1])
    if pathname[4] == "M":
        truth=1
    else:
        truth=0
   
    if correct == truth:
        counter+=1
      
    counter2+=1
    print("Inference for " + os.path.basename(image_filename) + " ---> " + "'" + infer_labels[(infer_probabilities.argsort()[1])] + "'")
    print('')
    print('Top results from most certain to least:')
    num_results = len(infer_labels)
    for index in range(0, num_results):
        one_prediction = '  certainty ' + str(infer_probabilities[index]) + ' --> ' + "'" + infer_labels[index]+ "'"
        print(one_prediction)
    print("time taken: "+str(checktime))
    print('-----------------------------------------------------------')
    
    
def main():
    
    global checktime
    t=0
    # get list of all the .png files in the image directory
    image_name_list = os.listdir(IMAGE_PATH)

    # filter out non .png files
    image_name_list = [IMAGE_PATH + '/' + a_filename for a_filename in image_name_list if a_filename.endswith('.png')]

    if (len(image_name_list) < 1):
        # no images to show
        print('No .png files in ' + IMAGE_PATH)
        return 1


    # initialize the neural compute device via the NCAPI
    device, graph = do_initialize()

    if (device == None or graph == None):
        print ("Could not initialize device.")
        quit(1)

    print(image_name_list)
    # loot through all the input images and run inferences and show results
    for index in range(0, len(image_name_list)):
       infer_labels, infer_probabilities = do_inference(graph, image_name_list[index])
    	#print(infer_labels,infer_probabilities)
       show_inference_results(image_name_list[index], infer_labels,infer_probabilities)
       t+=checktime
    print("Avg Time: ",t/counter2)
    print(counter," of ",counter2, " correct")
    #print(counter2)
    # clean up the NCAPI devices
    do_cleanup(device, graph)

main()
#if __name__ == "__main__":
#    sys.exit(main())
