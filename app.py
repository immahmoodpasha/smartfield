from flask import *
import requests
import joblib
import numpy as np
import os
import json
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from cropinfo import crop_info

# Initialize Flask application
app = Flask(__name__)

# Load the trained LightGBM model
model = joblib.load('lgbm_model.pkl')



@app.route('/')
def index():
    return render_template('crop.html')

# @app.route('/main') 
# def homie():
#     return render_template('homie.html')

@app.route('/home')
def homepage():
    return render_template('home.html')

#weather
@app.route('/weather',methods=['GET','POST'])
def weather():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city')
        if city:
            api_key = '4addbd52212f83ad2fbf5deba2f516e1'  # Replace with your actual API key from OpenWeatherMap
            url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    'city': data['name'],
                    'temperature': data['main']['temp'],
                    'description': data['weather'][0]['description'],
                    'icon': data['weather'][0]['icon']
                }
            else:
                weather_data = {'error': 'City not found'}
    return render_template('weather.html', weather=weather_data)



@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Convert data into numpy array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON response
    return render_template('result.html', prediction=prediction)


#crop disease prediction

model2=load_model('rice_leaf_disease_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Invert the class_indices dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

# Function to prepare an image for prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input
    img_array /= 255.0  # Normalize the image
    return img_array

def prepare_imagec(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/disease')
def home():
    return render_template('index.html')

@app.route('/diseasepredict', methods=['POST'])
def diseasepredict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Prepare the image
        img_array = prepare_image(file_path)
        
        # Make prediction
        prediction = model2.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        
        # Get class name from the predicted index
        result = class_names[predicted_class_index]
        
        return "prediction is :"+result

#cotton disease prediction
model3=load_model('modelx.h5')
@app.route('/cottondisease')
def cotton():
    return render_template('cotton.html')

@app.route('/cottonpredict', methods=['POST'])
def cottonpredict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Prepare the image
        img_array = prepare_imagec(file_path)
        class_labels = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy'] 
        # Make prediction
        prediction = model3.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        return "prediction is :"+predicted_class




if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
