from flask import Flask, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import tensorflow as tf 
import numpy as np
import json
import os
import string
import random
import classifier

#Global variables
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images'
app.config['LOST_FOLDER'] = './lostImages'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
modelFullPath = './output_graph.pb'
labelsFullPath = './output_labels.txt'

#function to check if uploaded file is in jpeg format
def allowedfile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""Creates a graph from saved GraphDef file and returns a saver."""
def create_graph():
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path) 

""" This function accepts an image and predicts its class """
@app.route('/lostPost', methods=['POST'])
def lostPost():

    client = MongoClient('mongodb://localhost:27017/')
    db = client.matcher
    collection = db.images    

    #Get the count of the db
    try:
        id = db.images.count()
    except Exception:
        return jsonify({"status":201, "message": "Failed to query database"})

    try:
        
        #upload image
        imagefile = request.files['imageFile']


        #check if the file id jpg
        if allowedfile(imagefile.filename) == False:
            return jsonify({"status":201, "message": "Invalid image"})


        imagename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)) + id.__str__()+'.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],imagename)
        imagefile.save(filepath)

        #feed the image to the classifier and return the result
        selected = classifier.run_inference_on_image(filepath,sess)

        f = open(labelsFullPath, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        selected_label = labels[selected]

        try:
            #store the new image and the label into database
            item = {"image": imagename, "prediction": selected, "claim":False }
            imageId = collection.insert_one(item).inserted_id
        except Exception as e:
            return jsonify({"status":200, "message": "Failed to query database"})
        
    except Exception as e:
        return jsonify({"status":201, "message": str(e)})

    return jsonify({"status":100, "message": "Success", "prediction": selected_label})

""" This function accepts an image and predicts its class """
@app.route('/foundPost', methods=['POST'])
def foundPost():

    client = MongoClient('mongodb://localhost:27017/')
    db = client.matcher
    collection = db.images

    try:
        
        #upload image
        imagefile = request.files['imageFile']

        #check if the file id jpg
        if allowedfile(imagefile.filename) == False:
            return jsonify({"status":201, "message": "Invalid image"})


        imagename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) +'.jpg'
        filepath = os.path.join(app.config['LOST_FOLDER'],imagename)
        imagefile.save(filepath)

        #feed the image to the classifier and return the result
        selected = classifier.run_inference_on_image(filepath,sess)

        f = open(labelsFullPath, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        selected_label = labels[selected]

        try:
            #store the new image and the label into database
            matched = []
            images = collection.find({"prediction": selected})
            for item in images:
                matched.append({"id": str(item['_id']), "image": item['image']})

        except Exception as e:
            return jsonify({"status":200, "message": "Failed to query database"})
        
    except Exception as e:
        return jsonify({"status":201, "message": str(e)})

    return jsonify({"status":100, "message": "Success", "prediction": selected_label, "matched": matched})

@app.route('/claimItem', methods=['POST'])
def claimItem():

    client = MongoClient('mongodb://localhost:27017/')
    db = client.matched
    collection = db.images

    try:
        imageId = request.form['id']
        collection.update_one({"_id" : ObjectId(imageId)}, {'$set' : {'claim': True}})
    except Exception as e:
        return jsonify({"status":200, "message": "Failed to query database"})
    
    return jsonify({"status":100, "message": "Success"})       

create_graph()
sess = tf.Session()

if __name__ == '__main__':
    port = 8080
    app.run(host='127.0.0.1', port=port)
