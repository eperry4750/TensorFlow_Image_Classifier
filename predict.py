import numpy as np
import json

import tensorflow as tf
import tensorflow_hub as hub

from argparse import ArgumentParser
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


def process_image(image):
    IMAGE_SIZE = 224
    image_tensor = tf.convert_to_tensor(image)
    resized_image = tf.image.resize(image_tensor, (IMAGE_SIZE, IMAGE_SIZE))/255.0
    return resized_image.numpy()


def predict(image_path, model_path, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    finished_image = np.expand_dims(processed_image, axis=0)
    predictions = model_path.predict(finished_image)
    top_predictions = tf.math.top_k(predictions, k = top_k)
    return top_predictions[0].numpy(), top_predictions[1].numpy()

    
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, type=str,
                        help="Path to a trained model.")
    parser.add_argument("-i", "--image_path", required=True, type=str,
                        help="Path to an image file")
    parser.add_argument("-k", "--top_k", required=False, type=int,
                        default=1,
                        help="Return top k probabiliites.")
    parser.add_argument("-l", "--label_map", required=False, type=str, default=None,
                        help="A mapping of classes to real names from a json file")
    
    return parser 


def predict_main(args):
    
    model_path = args.model_path
    saved_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    image_path = args.image_path
    top_k = args.top_k
    label_map = args.label_map
    
    probs, classes = predict(image_path, saved_model, top_k)
        
    if label_map == None:
        for prob, clas in zip(probs[0], classes[0]):
            print(prob, clas)
    else:
        with open(label_map, 'r') as f:
            class_names = json.load(f)
            for prob, clas in zip(probs[0], [class_names[str(clas + 1)] for clas in classes[0]]):
               print(prob, clas)

    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    predict_main(args)


if __name__ == '__main__':
    main()    