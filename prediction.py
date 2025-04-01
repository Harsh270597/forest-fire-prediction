import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def generate_sample_data(n_samples=34):  # Increased to 34
    np.random.seed(42)
    temperature = np.random.uniform(20, 45, n_samples)
    humidity = np.random.uniform(10, 90, n_samples)
    wind_speed = np.random.uniform(0, 30, n_samples)
    rainfall = np.random.uniform(0, 50, n_samples)
    
    risk_score = 0.4 * temperature - 0.3 * humidity + 0.2 * wind_speed - 0.1 * rainfall
    risk_score = (risk_score - np.min(risk_score)) / (np.max(risk_score) - np.min(risk_score))
    
    risk_level = np.zeros(n_samples, dtype=object)
    risk_level[(risk_score >= 0) & (risk_score < 0.25)] = 'Low'
    risk_level[(risk_score >= 0.25) & (risk_score < 0.5)] = 'Moderate'
    risk_level[(risk_score >= 0.5) & (risk_score < 0.75)] = 'High'
    risk_level[(risk_score >= 0.75)] = 'Severe'
    
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'risk_level': risk_level
    })
    
    return data

class ForestFireDataset(Dataset):
    def __init__(self, features, targets, seq_length=3):  # Reduced from 7 to 3
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return max(0, len(self.features) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_batches = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_batches += 1
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        train_loss = running_loss / max(1, train_batches)
        val_loss = val_loss / max(1, val_batches)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def predict_risk(model, features, seq_length, feature_scaler, target_scaler, risk_levels):
    model.eval()
    with torch.no_grad():
        seq = features[-seq_length:].reshape(1, seq_length, -1)
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

def main():
    data = generate_sample_data(34)
    print("Sample data generated:")
    print(data.head())
    
    risk_mapping = {'Low': 0, 'Moderate': 1/3, 'High': 2/3, 'Severe': 1}
    data['risk_numeric'] = data['risk_level'].map(risk_mapping)
    
    seq_length = 3  # Reduced from 7 to 3
    features = data[['temperature', 'humidity', 'wind_speed', 'rainfall']].values
    targets = data[['risk_numeric']].values
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets)
    
    if len(features) <= seq_length:
        raise ValueError("Not enough data points for the specified sequence length")
    
    # Changed test_size from 0.2 to 0.1
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.1, random_state=42)
    
    train_dataset = ForestFireDataset(X_train, y_train, seq_length)
    val_dataset = ForestFireDataset(X_val, y_val, seq_length)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is too small after sequence processing")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    input_size = 4
    hidden_size = 32
    num_layers = 1
    output_size = 1
    
    model = ForestFireLSTM(input_size, hidden_size, num_layers, output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining the model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    
    sample_input = np.array([[40, 20, 25, 0]])
    sample_input_scaled = feature_scaler.transform(sample_input)
    sample_sequence = np.tile(sample_input_scaled, (seq_length, 1))
    
    risk_level, risk_score = predict_risk(
        model, sample_sequence, seq_length, feature_scaler, target_scaler, 
        list(risk_mapping.keys())
    )
    
    print(f"\nSample Prediction:")
    print(f"Input: Temperature = {sample_input[0,0]}°C, Humidity = {sample_input[0,1]}%, Wind Speed = {sample_input[0,2]} km/h, Rainfall = {sample_input[0,3]} mm")
    print(f"Predicted Risk Level: {risk_level} (Score: {risk_score:.4f})")
    
    torch.save(model.state_dict(), 'forest_fire_lstm_model.pth')
    print("\nModel saved as 'forest_fire_lstm_model.pth'")

if __name__ == "__main__":
    main()