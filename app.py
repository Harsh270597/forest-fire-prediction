import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the LSTM model class
class ForestFireLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ForestFireLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
feature_scaler = None
target_scaler = None
seq_length = 3

def load_model():
    """Load the trained model and scalers"""
    global model, feature_scaler, target_scaler
    
    try:
        # Initialize model with the same architecture as training
        input_size = 4
        hidden_size = 32
        num_layers = 1
        output_size = 1
        
        model = ForestFireLSTM(input_size, hidden_size, num_layers, output_size)
        
        # Load the trained weights
        model_path = 'D:\\ML\\forest_fire_final\\forest_fire_lstm_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load scalers
        scaler_path = 'scalers.json'
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'r') as f:
                    scaler_data = json.load(f)
                
                # Feature scaler
                feature_scaler = MinMaxScaler()
                feature_scaler.min_ = np.array(scaler_data['feature_scaler']['min'])
                feature_scaler.scale_ = np.array(scaler_data['feature_scaler']['scale'])
                feature_scaler.data_min_ = np.array(scaler_data['feature_scaler']['data_min'])
                feature_scaler.data_max_ = np.array(scaler_data['feature_scaler']['data_max'])
                feature_scaler.data_range_ = np.array(scaler_data['feature_scaler']['data_range'])
                
                # Target scaler
                target_scaler = MinMaxScaler()
                target_scaler.min_ = np.array(scaler_data['target_scaler']['min'])
                target_scaler.scale_ = np.array(scaler_data['target_scaler']['scale'])
                target_scaler.data_min_ = np.array(scaler_data['target_scaler']['data_min'])
                target_scaler.data_max_ = np.array(scaler_data['target_scaler']['data_max'])
                target_scaler.data_range_ = np.array(scaler_data['target_scaler']['data_range'])
                
                logger.info("Scalers loaded successfully")
            except Exception as e:
                logger.error(f"Error loading scalers: {e}")
                raise
        else:
            logger.warning("Scaler file not found. Using default scalers")
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(np.array([[20, 10, 0, 0], [45, 90, 30, 50]]))
            
            target_scaler = MinMaxScaler()
            target_scaler.fit(np.array([[0], [1]]))
            
    except Exception as e:
        logger.error(f"Failed to load model or scalers: {e}")
        raise

def predict_risk(features):
    """Make a prediction using the loaded model"""
    try:
        model.eval()
        with torch.no_grad():
            features_scaled = feature_scaler.transform(features)
            seq = np.tile(features_scaled, (seq_length, 1))
            seq = seq.reshape(1, seq_length, -1)
            
            seq_tensor = torch.FloatTensor(seq)
            prediction = model(seq_tensor).numpy()
            
            prediction = target_scaler.inverse_transform(prediction)
            
            risk_score = prediction[0, 0]
            if risk_score < 0.25:
                risk = 'Low'
            elif risk_score < 0.5:
                risk = 'Moderate'
            elif risk_score < 0.75:
                risk = 'High'
            else:
                risk = 'Severe'
                
            return risk, risk_score
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

@app.route('/')
def home():
    """Render the home page with the input form"""
    if not os.path.exists('templates/index.html'):
        return "Template not found. Please run create_templates endpoint first.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    """Handle prediction requests"""
    try:
        data = request.get_json() if request.is_json else request.form
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        wind_speed = float(data.get('wind_speed', 0))
        rainfall = float(data.get('rainfall', 0))
        
        input_data = np.array([[temperature, humidity, wind_speed, rainfall]])
        risk_level, risk_score = predict_risk(input_data)
        
        response = {
            'risk_level': risk_level,
            'risk_score': float(risk_score),
            'input': {
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'rainfall': rainfall
            }
        }
        logger.info(f"Prediction made: {response}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/create_templates', methods=['GET'])
def create_templates():
    """Create the HTML templates if they don't exist"""
    try:
        os.makedirs('templates', exist_ok=True)
        template_path = 'templates/index.html'
        
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write('''[Same HTML content as before - omitted for brevity]''')
            logger.info("Templates created successfully")
            return "Templates created successfully!", 200
        else:
            logger.info("Templates already exist")
            return "Templates already exist", 200
    except Exception as e:
        logger.error(f"Error creating templates: {e}")
        return f"Error creating templates: {e}", 500

def save_scalers():
    """Save the scaler parameters to a JSON file"""
    try:
        if feature_scaler is None or target_scaler is None:
            logger.warning("Scalers not initialized, cannot save")
            return
        
        scaler_data = {
            'feature_scaler': {
                'min': feature_scaler.min_.tolist(),
                'scale': feature_scaler.scale_.tolist(),
                'data_min': feature_scaler.data_min_.tolist(),
                'data_max': feature_scaler.data_max_.tolist(),
                'data_range': feature_scaler.data_range_.tolist()
            },
            'target_scaler': {
                'min': target_scaler.min_.tolist(),
                'scale': target_scaler.scale_.tolist(),
                'data_min': target_scaler.data_min_.tolist(),
                'data_max': target_scaler.data_max_.tolist(),
                'data_range': target_scaler.data_range_.tolist()
            }
        }
        
        with open('scalers.json', 'w') as f:
            json.dump(scaler_data, f)
        logger.info("Scalers saved successfully")
    except Exception as e:
        logger.error(f"Error saving scalers: {e}")

if __name__ == '__main__':
    try:
        load_model()
        save_scalers()
    except Exception as e:
        logger.error(f"Startup error: {e}")
        print(f"Startup error: {e}")
        exit(1)
    
    # Create templates if they don't exist
    create_templates()
    
    # Start Flask app on static port
    port = 5000
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)