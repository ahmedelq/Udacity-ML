import argparse
from pathlib import Path
import json 
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def process_image(img):
    img = tf.cast(img, tf.float32) #change type
    img = tf.image.resize(img, (224, 224)) #resize
    img /= 225 #adjust pixel values
    return img.numpy()

def predict(image_path, model, top_k):
    #img = tf.io.read_file(image_path)
    #img = tf.image.decode_jpeg(img, channels=3)
    img = np.asarray(Image.open(image_path))
    img = process_image(img)
    img = tf.expand_dims(img, axis=0) #adding extra dimension
    classes = model.predict(img)
    class_prob = list(zip(
                          range(len(classes[0])), 
                          classes[0]))
    tops  = sorted(class_prob, key= lambda x: x[1], reverse=True)[:top_k]
    labels = [str(lbl+1) for lbl,_ in tops]
    probs = [p for _, p in tops]
    return probs, labels

def cat_name_construct(path):
    if not path:
        return lambda x:x 
    with path.open() as file:
        contents = json.load(file)
    return lambda x: contents.get(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='102', description='Flower classifier.')

    parser.add_argument('img_path', metavar='IMG_PATH', type=str,
                    help='The input image to be classified')

    parser.add_argument('model_path', metavar='MODEL_PATH', type=str,
                    help='The path of the model to be used.')

    parser.add_argument('--top_k',  metavar='N', type=int, help="Show the top N classes for the classified image.")
    parser.add_argument('--category_names', metavar='map.json', type=Path, help="Load labels from a json mapping.")

    args = parser.parse_args()
    #print('Image path:', args.img_path)
    #print('Model:', args.model_path)
    try:
        cat_names = cat_name_construct(args.category_names)
        top_k = args.top_k if args.top_k else 1
        if (top_k <= 0):
            exit("Invalid input for top_k.")
        saved_model = tf.keras.models.load_model(args.model_path,
                                            custom_objects={'KerasLayer': hub.KerasLayer})
        p,l = predict(args.img_path, saved_model, top_k)
        l = [cat_names(lbl) for lbl in l]
        pl = list(zip(p,l))[:top_k]
        print("\n".join( 
            ["Class {}: {}".format(lbl, prop)  for prop,lbl in pl ] )
            )

    except Exception as e:
        exit("Some fetal error occured: " +  e.__str__())

