import json
import os
from pathlib import Path


class Saver:
    def __init__(self):
        pass

    @staticmethod
    def save_model(model, file_model_JSON, file_weights):
        """General save model to disk function """

        if Path(file_model_JSON).is_file():
            os.remove(file_model_JSON)
        json_string = model.to_json()

        with open(file_model_JSON, 'w') as f:
            json.dump(json_string, f)

        if Path(file_weights).is_file():
            os.remove(file_weights)

        model.save_weights(file_weights)