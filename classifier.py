import os
import numpy as np
import tensorflow as tf
import json

modelFullPath = './output_graph.pb'
lables = ['Indica','Sativa','Hybrid']
    

def run_inference_on_image(img, sess):
    
    if not tf.gfile.Exists(img):
        tf.logging.fatal('File does not exist %s', img)
        return ""

    image_data = tf.gfile.FastGFile(img, 'rb').read()
        
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})                   
                    
    predictions = np.squeeze(predictions)

    return np.asscalar(np.argmax(predictions))