from keras.models import model_from_json
from time import time
from os import path, makedirs


def model_to_json(model, filename):
    """
    Serialize model to JSON

    @param filename:      string
    @param model:       keras.models
    @return:
    """
    timestamp = str(time()).split('.')[0]
    folder = "models/" + filename + '-' + timestamp + '/'
    if not path.exists(folder):
        makedirs(folder)

    model_json = model.to_json()
    with open(folder + filename + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(folder + filename + ".h5")
    print("Keras model saved.")


def json_to_model(filepath):
    """
    Load json and create model. Model definition and its weight must have same filename.

    @param filepath:    string - json model definition
    @return:            keras.models
    """
    json_file = open(filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path.splitext(filepath)[0] + '.h5')
    print("Model loaded")
    return loaded_model
