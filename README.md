<div align="center">
<img src="images/logo.png" alt="logo">

# Kealzyl Caleum
**The least reliable AI weather forecast**

</div>

<br>

### ! You can find the pretrained model on [HuggingFace](https://huggingface.co/Kealzyl/caleum) trained on Romania/Bucharest !

**Caleum** is an experimental AI-powered weather prediction model developed in Python using PyTorch and GPU-accelerated CUDA training. It learns patterns from historical weather observations to predict upcoming atmospheric conditions.

Using a two-layer Gated Recurrent Unit (GRU) neural network, Caleum analyzes the previous 72 hours of weather data to generate experimental forecasts for the next 24 hours.

### Features

- **CUDA-Accelerated Training:** Uses NVIDIA GPUs through PyTorch CUDA when available, with automatic CPU fallback.
- **Real Weather Data:** Trains on historical hourly observations obtained from the Open-Meteo Historical Weather API.
- **24-Hour Predictions:** Forecasts upcoming weather conditions using recent observations.
- **Multiple Weather Variables:** Predicts temperature, relative humidity, surface pressure, wind speed, and precipitation.
- **Model Checkpointing:** Saves the best-performing model based on validation loss.
- **Forecast Export:** Exports predictions to CSV for further analysis and visualization.
- **Chronological Validation:** Evaluates the model on later data to help measure its ability to generalize to unseen weather observations.

### Technology Stack

- Python
- PyTorch
- NVIDIA CUDA
- NumPy
- Pandas
- Open-Meteo API

### Project Goal

Caleum aims to explore how deep learning and GPU computing can be applied to weather forecasting. By learning temporal patterns in meteorological data, the project provides a foundation for developing more advanced prediction systems.

**Note:** Caleum is an experimental forecasting model, not a replacement for professional meteorological systems. Its accuracy depends on the quality of its training data, model architecture, and validation results.

