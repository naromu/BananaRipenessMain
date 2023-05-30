import base64
import numpy as np
import os
from io import BytesIO
from flask import Flask, request
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from pymongo.mongo_client import MongoClient
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception

class ImageClassifierApp:
    
    def __init__(self):
        self.app = Flask(__name__, static_folder='./static')
        
        CORS(self.app)
        self.model_densenet121 = load_model('modelos/ModelBananaRipenessClassification_densenet121.h5')
        self.model_mobilenetv2 = load_model('modelos/ModelBananaRipenessClassification_mobilenetv2.h5')
        self.model_resnet50v2 = load_model('modelos/ModelBananaRipenessClassification_resnet.h5')
        self.model_xception = load_model('modelos/ModelBananaRipenessClassification_xception.h5')
        self.models = {
             'resnet50v2': {
             'model':self.model_resnet50v2
            },
            'densenet121': {
            'model': self.model_densenet121
            },
            'mobilenetv2': {
            'model': self.model_mobilenetv2
            },
            'xception': {
            'model': self.model_xception
            }
        }

        self.string_base64 = None
        self.label = "Toma una foto de un banano"
        self.base_image_path = "/static/BananaRipenessClassifierImage.jpeg"
        self.img_buffer = None
        self.image_counter = 0
         
        self.app.add_url_rule("/", "index", self.index, methods=["GET"])
        self.app.add_url_rule("/prediction", "prediction", self.prediction, methods=["POST"])

    def connection_mongoDB(self):
        password = os.environ.get('MONGODB_PASSWORD')
        uri = f"mongodb+srv://Bananas:{password}@bananaripenesscluster.ggz0sia.mongodb.net/"
        client = MongoClient(uri)
    
        try:
            client.admin.command('ping')
            db = client["Banana_Ripeness_DB"]
            collection = db["Banana_Ripeness_Collection"]
      
            result = collection.find_one({}, {"bananas": 1})
       
            if result:
                self.bananas = result["bananas"]
                class_names = [banana["class_name"] for banana in self.bananas]
                print("Los nombres de las clases son:", class_names)
            else:
                print("No se encontraron datos de bananos en la colección.")

            self.class_names = class_names

        except Exception as e:
            print("El error es:",e)

    def majority_voting(self, models):
        num_classes = 3
        votes = np.zeros(num_classes)
        confidence = np.zeros(num_classes)

        for model_name, model_data in models.items():
            prediction = model_data['prediction']
            class_prediction = model_data['class_prediction']
            confidence_score = model_data['confidence_score']
            print(f"Modelo: {model_name}\t=>\tpredicción: {prediction} - {class_prediction}\t=>\tconfianza: {confidence_score}")
            votes[prediction] += 1
            confidence[prediction] += confidence_score

        # len(set(votes)) y len(votes) son iguales es que ninguna clase tuvo la misma cantidad de votos, es decir que no hubo empate
        if len(set(votes)) == len(votes):
            return np.argmax(votes)
        else:
            return np.argmax(confidence)

    def prediction(self):

        self.connection_mongoDB()

        json_data = request.json
        self.string_base64 = json_data["string_base"]
        img_data = base64.b64decode(self.string_base64)
        self.img_buffer = BytesIO(img_data)
        
        img = image.load_img(self.img_buffer, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x_resnet = preprocess_resnet(x.copy())
        x_densenet = preprocess_densenet(x.copy())
        x_mobilenet = preprocess_mobilenet(x.copy())
        x_xception = preprocess_xception(x.copy())
        #print("\nLos nombres de las clases son:", self.class_names, "\n")
        
        for model_name, model_data in self.models.items():
            if model_name == 'resnet50v2':
                prediction_scores = model_data['model'].predict(x_resnet)[0]
            elif model_name == 'densenet121':
                prediction_scores = model_data['model'].predict(x_densenet)[0]
            elif model_name == 'mobilenetv2':
                prediction_scores = model_data['model'].predict(x_mobilenet)[0]
            elif model_name == 'xception':
                prediction_scores = model_data['model'].predict(x_xception)[0]
            else:
                raise ValueError(f"Unrecognized model name '{model_name}'")
            
            prediction = np.argmax(prediction_scores)
            model_data['prediction'] = prediction
            model_data['class_prediction'] = self.class_names[prediction]
            model_data['confidence_score'] = prediction_scores[prediction]
        print("\n")
        best_prediction = self.majority_voting(self.models)

        print("\nMejor predicción:", best_prediction)

        predicted_class_name = self.class_names[best_prediction]
        print()
        print("\nLa clase predicha es:", predicted_class_name)

        message = ""
        price = 0.0
        for banana in self.bananas:
            if banana["class_name"] == predicted_class_name:
                price = banana["price"]
                message = banana["message"]
                break
        print("\nel precio del banano es:", price)
        print("\nel mensaje del banano es:", message)
        self.label = predicted_class_name
        return {"data": self.label,"mensaje": message, "precio":price} 


    def index(self):
        if self.string_base64 is not None:
            imgstring = "data:image/jpeg;base64," + self.string_base64
        else:
            imgstring = self.base_image_path
            
        html = f'''
            <div style="display: flex; flex-direction: column; align-items: center; height: 75vh;">
                <img src="{imgstring}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />
                <p style="text-align: center; font-size: 72px; margin-top: 20px;">{self.label}</p>
            </div>
        '''
        return html

    def run(self, host="0.0.0.0", port=5000):
        self.connection_mongoDB() 
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    app = ImageClassifierApp()
    app.run()
