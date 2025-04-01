import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const ForestFirePredictionApp());
}

class ForestFirePredictionApp extends StatelessWidget {
  const ForestFirePredictionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Forest Fire Prediction',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _temperatureController = TextEditingController(text: '30');
  final _humidityController = TextEditingController(text: '40');
  final _windSpeedController = TextEditingController(text: '15');
  final _rainfallController = TextEditingController(text: '5');

  String _riskLevel = '';
  double _riskScore = 0.0;
  bool _isLoading = false;
  String _errorMessage = '';

  // Replace with your server IP
  final String _apiUrl = 'http://127.0.0.1:5000/predict';

  Future<void> _predictFireRisk() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = '';
      _riskLevel = '';
      _riskScore = 0.0;
    });

    try {
      final response = await http.post(
        Uri.parse(_apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'temperature': double.parse(_temperatureController.text),
          'humidity': double.parse(_humidityController.text),
          'wind_speed': double.parse(_windSpeedController.text),
          'rainfall': double.parse(_rainfallController.text),
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _riskLevel = data['risk_level'];
          _riskScore = data['risk_score'].toDouble();
        });
      } else {
        final errorData = json.decode(response.body);
        setState(() {
          _errorMessage = errorData['error'] ?? 'Unknown error occurred';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to connect to server: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Color _getRiskColor(String riskLevel) {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return Colors.green;
      case 'moderate':
        return Colors.orange;
      case 'high':
        return Colors.deepOrange;
      case 'severe':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Forest Fire Risk Prediction'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextFormField(
                controller: _temperatureController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Temperature (°C)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => _validateInput(value, 0, 60),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _humidityController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Humidity (%)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => _validateInput(value, 0, 100),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _windSpeedController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Wind Speed (km/h)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => _validateInput(value, 0, 100),
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _rainfallController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Rainfall (mm)',
                  border: OutlineInputBorder(),
                ),
                validator: (value) => _validateInput(value, 0, 100),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _isLoading ? null : _predictFireRisk,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? const CircularProgressIndicator(color: Colors.white)
                    : const Text('Predict Fire Risk', style: TextStyle(fontSize: 18)),
              ),
              const SizedBox(height: 24),
              if (_riskLevel.isNotEmpty)
                Card(
                  elevation: 4,
                  color: _getRiskColor(_riskLevel),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Prediction Result',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Risk Level: $_riskLevel',
                          style: const TextStyle(fontSize: 16, color: Colors.white),
                        ),
                        Text(
                          'Risk Score: ${_riskScore.toStringAsFixed(2)}',
                          style: const TextStyle(fontSize: 16, color: Colors.white),
                        ),
                      ],
                    ),
                  ),
                ),
              if (_errorMessage.isNotEmpty)
                Card(
                  elevation: 4,
                  color: Colors.red[100],
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Text(
                      'Error: $_errorMessage',
                      style: TextStyle(color: Colors.red[900]),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  String? _validateInput(String? value, double min, double max) {
    if (value == null || value.isEmpty) {
      return 'Please enter a value';
    }
    final numValue = double.tryParse(value);
    if (numValue == null) {
      return 'Please enter a valid number';
    }
    if (numValue < min || numValue > max) {
      return 'Value must be between $min and $max';
    }
    return null;
  }

  @override
  void dispose() {
    _temperatureController.dispose();
    _humidityController.dispose();
    _windSpeedController.dispose();
    _rainfallController.dispose();
    super.dispose();
  }
}