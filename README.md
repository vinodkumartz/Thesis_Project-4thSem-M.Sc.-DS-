
# Wi-Fi Router Placement Optimization Using Deep Learning

## Project Overview
This project implements a deep learning-based approach to optimize router placement in indoor Wi-Fi networks by analyzing Received Signal Strength Indicator (RSSI) data. It builds upon the thesis titled *"Enhancing Signal Coverage and Performance by Optimization of Router Placement Using Deep Learning Model"* by Vinod Kumar, submitted for an M.Sc. in Data Science at the Indian Institute of Information Technology, Lucknow (2023-2025).

The goal is to predict optimal router positions to maximize signal coverage and performance in indoor environments using advanced deep learning models, including Bidirectional LSTM, GRU, LSTM with Attention, and Conv1D + Bidirectional LSTM. The project leverages two datasets: a reference dataset from literature and a real-world dataset collected in a controlled lab environment.

## Thesis Project Report
- **Report Link** : https://drive.google.com/file/d/1B_xsJGdpMvaEudf_HR36pxxOvtLpNG_x/view?usp=sharing

## Features
- **Deep Learning Models**: Implements Bidirectional LSTM, GRU, LSTM with Attention, and Conv1D + Bidirectional LSTM for RSSI-based localization.
- **Datasets**:
  - **Reference Dataset**: 194 samples with RSSI values from 4 access points, reshaped to (194, 4, 1).
  - **Real-World Dataset**: 364 samples with RSSI values from 6 access points, reshaped to (364, 3, 2).
  - **External Test Set**: 7 samples for validation, reshaped to (7, 3, 2).
- **Preprocessing**: Handles missing values, normalizes RSSI data, and reshapes inputs for model compatibility.
- **Evaluation Metrics**: Includes MSE, RMSE, accuracy, precision, recall, and F1-score with a 6.5-meter threshold.
- **Visualization**: Plots training/validation loss, true vs. predicted coordinates, and highlights optimal router placements.

## Prerequisites
- Python 3.8+
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow/Keras
  - Seaborn
  - Matplotlib
- Tools:
  - NetSpot RSSI value analyzer (for real-world data collection)
  - Jupyter Notebook (optional, for interactive development)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure datasets (`reference_dataset.csv`, `real_world_dataset.csv`, `external_test_set.csv`) are placed in the `data/` directory.

## Usage
1. **Data Preparation**:
   - Place the datasets in the `data/` folder.
   - Update file paths in the script if necessary.

2. **Running the Code**:
   - Execute the main script to train and evaluate models:
     ```bash
     python main.py
     ```
   - The script performs:
     - Data loading and preprocessing
     - Model training (Bidirectional LSTM, GRU, LSTM with Attention, Conv1D + Bi-LSTM)
     - Evaluation on test sets
     - Visualization of results (loss curves, coordinate scatter plots)

3. **Model Configurations**:
   - Bidirectional LSTM: 64 units, dropout, dense layers (32, 2 units).
   - GRU: 64 units, dropout, dense layers (32, 2 units).
   - LSTM with Attention: 64-unit LSTM, attention layer, dense layers (64, 2 units).
   - Conv1D + Bi-LSTM: 64-filter Conv1D, 64-unit Bi-LSTM, 32-unit LSTM, dense layers (64, 3 units).
   - Training parameters: Adam optimizer (lr=0.001), MSE loss, 100 epochs, batch size=32.

4. **Output**:
   - Trained models saved in `models/`.
   - Visualizations (loss curves, coordinate plots) saved in `outputs/`.
   - Evaluation metrics printed to console and saved in `results/`.

## Dataset Details
- **Reference Dataset**: From Alhmiedat (2023), contains 194 samples with RSSI from 4 access points, covering a 20m x 6m area.
- **Real-World Dataset**: Collected using NetSpot with 6 access points, 364 samples, covering a 10.058m x 12.192m x 4.572m area.
- **Preprocessing**:
  - Missing values replaced with -100 dBm or mean imputation.
  - RSSI normalized to [0, 1] using Min-Max scaling.
  - Data reshaped for model input (e.g., (samples, timesteps, features)).

## Results
- **Reference Dataset**:
  - Bidirectional LSTM and LSTM with Attention: 97.44% accuracy, 0.99 F1-score.
  - Outperforms baseline Improved RNN (93.25% accuracy).
- **Real-World Dataset**:
  - GRU and LSTM with Attention: 98.63% accuracy, 0.99 F1-score.
  - Conv1D + Bi-LSTM and Bi-LSTM: 95.89% accuracy.
- **External Test Set**: Accurate coordinate predictions, with GRU and LSTM with Attention showing robust generalization.

## Project Structure
```
├── data/
│   ├── reference_dataset.csv
│   ├── real_world_dataset.csv
│   ├── external_test_set.csv
├── models/
│   ├── bidirectional_lstm.h5
│   ├── gru.h5
│   ├── lstm_attention.h5
│   ├── conv1d_bilstm.h5
├── outputs/
│   ├── loss_curves/
│   ├── coordinate_plots/
├── results/
│   ├── evaluation_metrics.txt
├── main.py
├── requirements.txt
├── README.md
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.

## Acknowledgments
- Supervisors: Dr. Deepak Kumar Singh and Dr. Madhurima Datta.
- Mentor: Mr. Sourav Raj, Data Scientist at Jio Platform Limited.
- Support: Department of Mathematics, Indian Institute of Information Technology, Lucknow.

## License
This project is licensed under the MIT License.

## Contact
For inquiries, contact Vinod Kumar  at Indian Institute of Information Technology, Lucknow.
- Eamil : imvinodkumarr@gmail.com
