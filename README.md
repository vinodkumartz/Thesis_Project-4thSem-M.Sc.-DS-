
# Wi-Fi Router Placement Optimization Using Deep Learning

## Project Overview
This project implements a deep learning-based approach to optimize router placement in indoor Wi-Fi networks by analyzing Received Signal Strength Indicator (RSSI) data. It builds upon the thesis titled *"Enhancing Signal Coverage and Performance by Optimization of Router Placement Using Deep Learning Model"* by Vinod Kumar, submitted for an M.Sc. in Data Science at the Indian Institute of Information Technology, Lucknow (2023-2025).

The goal is to predict optimal router positions to maximize signal coverage and performance in indoor environments using advanced deep learning models, including Bidirectional LSTM, GRU, LSTM with Attention, and Conv1D + Bidirectional LSTM. The project leverages two datasets: a reference dataset from literature and a real-world dataset collected in a controlled lab environment.

## Thesis Project Report
- **Report Link** : https://drive.google.com/file/d/1B_xsJGdpMvaEudf_HR36pxxOvtLpNG_x/view?usp=sharing

## Abstract
Accurate localization of devices in indoor environments remains a challenging problem
due to signal fluctuations caused by obstacles and dynamic movement of nodes (e.g., cus-
tomer devices). The present study builds on existing work focused on optimizing router
placement in Wi-Fi networks using Received Signal Strength Indicator (RSSI) values. In
previous studies that utilized an improved RNN with LSTM layers for spatial analysis,
these methods have limitations in a single model without exploring broader deep learning
alternatives.
In the present study, the reference work is extended by experimenting with a diverse
range of advanced deep learning models on the same WiFi RSS fingerprint dataset. We
evaluated multiple architectures, including CNN, Bidirectional LSTM, Stacked LSTM,
GRU, LSTM with Attention, Conv1D-LSTM, Temporal Convolutional Networks (TCN),
and Transformer networks. In the present investigation, an optimum accuracy of 97.44%
is achieved by incorporating the LSTM with Attention, Bi-LSTM models, also providing
highly accurate results. The present study demonstrates improved results to the original
model(improved RNN), demonstrating better adaptability to spatial and temporal varia-
tions in indoor signal propagation.
To validate the robustness of the proposed methodology, a new dataset was collected
following a standardized data collection process derived from the literature survey. The
same set of deep learning models was trained on this dataset and observed consistent per-
formance improvements, with GRU and LSTM-Attention models achieving up to 98.63%
accuracy, along with high precision, recall, and F1 scores.
Overall, our research introduces a more comprehensive and comparative deep learning
models. This work can assist network engineers and homeowners in designing efficient
Wi-Fi layouts while contributing to the field of spatial signal modeling and wireless net-
work optimization.

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

## Results
- **Reference Dataset**:
  - Bidirectional LSTM and LSTM with Attention: 97.44% accuracy, 0.99 F1-score.
  - Outperforms baseline Improved RNN (93.25% accuracy).
- **Real-World Dataset**:
  - GRU and LSTM with Attention: 98.63% accuracy, 0.99 F1-score.
  - Conv1D + Bi-LSTM and Bi-LSTM: 95.89% accuracy.
- **External Test Set**: Accurate coordinate predictions, with GRU and LSTM with Attention showing robust generalization.

## Conclusion 
This research investigated the use of advanced deep learning models for indoor localiza-
tion and optimal router placement based on Wi-Fi RSSI data. Deep learning architectures
were implemented and evaluated using both a publicly available reference dataset and a
collected real-world dataset in a laboratory environment. The experimental results pro-
vide important insights into the effectiveness and adaptability of various models under
different data conditions.
On the reference dataset, the Bidirectional LSTM and LSTM with Attention mod-
els achieved the highest performance, with each reaching an accuracy of 97.44% and F1
scores of 0.99. These outcomes represent a notable improvement compared to the base-
line Improved RNN, which recorded 93.25% accuracy. The findings highlight the advan-
tages of integrating bidirectional processing and attention mechanisms, which enhance the
model’s ability to capture spatial dependencies and accurately determine optimal router
placements.
On the real-world collected dataset, the GRU and LSTM with Attention models
achieved the highest performance, each reaching an accuracy of 98.63% and F1 scores
close to 1.00. These models exhibited a notable improvement over the baseline Improved
RNN, which recorded an accuracy of 93.25%. The findings highlight the effectiveness
of temporal sequence learning in GRU and the attention mechanism’s ability to enhance
focus on relevant features, enabling robust prediction even under complex indoor condi-
tions.
In summary, the advanced deep learning architectures—namely GRU, Bidirec-
tional LSTM, and LSTM with Attention—consistently demonstrated superior perfor-
mance compared to the Improved RNN across both the reference and real-world datasets.
Their strong generalization on structured data and robustness under real-world variability
highlight their effectiveness for deployment in smart indoor environments. These models
present viable solutions for key applications such as router positioning, signal enhance-
ment, and user localization, paving the way for more efficient and intelligent wireless
network systems.
Despite these achievements, there remain some research gaps and opportunities for
future work. The current study assumes a static environment without accounting for dy-
namic elements like moving objects, people, or changing layouts, which can influence sig-
39nal propagation. Additionally, the dataset size—especially the real-world samples—was
relatively small, which may limit generalizability to larger-scale environments. Future re-
search could experiment with the integration of reinforcement learning for adaptive router
repositioning, expand the dataset with more spatial diversity, and investigate hybrid mod-
els combining deep learning with physical signal propagation models. Real-time imple-
mentation and deployment in live environments would also validate the feasibility of these
models beyond offline simulations.

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
