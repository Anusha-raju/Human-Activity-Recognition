# Code Directory Overview

This folder contains all the core code related to training, prediction, and post-training analysis for the Human Activity Recognition (HAR) system. The system utilizes a **Long-term Recurrent Convolutional Network (LRCN)** to classify activities in videos.

## Folder Structure

```plaintext
code/
├── component/                # Core modules for training and prediction logic
│   ├── utils/                # Preprocessing and common code modules
│   │   ├── dataset.py        # Code to load and process the data
│   │   ├── utility.py        # Common functions for various tasks
│   │   ├── visualization.py  # Code to generate plot metrics
│   ├── model.py              # Model definition (LRCN)
│   ├── post_training.py      # Code for post-training analysis
│   ├── predict.py            # Code to handle model predictions
│   ├── train.py              # Code to handle model training
├── maincode/                 # Main execution scripts and model outputs
│   ├── models/               # Folder for storing saved trained models
│   ├── outputs/              # Folder for storing evaluation results
│   ├── main.py               # Main entry script to run training, evaluation, and analysis
│   ├── train.log             # Training logs
```

## Folders Overview

### `component/`

- **`utils/`**
  
  - `dataset.py`: Responsible for loading and processing the video dataset (e.g., UCF50).
    
  - `utility.py`: Contains helper functions for data preprocessing, model evaluation, and more.
    
  - `visualization.py`: Functions to generate visualizations, including training plots and metrics.
    
- **`model.py`**: Defines the architecture of the LRCN model (combining CNN and LSTM layers).
  
- **`post_training.py`**: Contains functions for performing post-training analysis such as generating heatmaps or activity labels.
  
- **`predict.py`**: Handles making predictions on input video clips using the trained model.
  
- **`train.py`**: Contains the code for training the LRCN model.
  

### `maincode/`

- **`models/`**: Stores saved trained models.
  
- **`outputs/`**: Contains evaluation results.
  
- **`main.py`**: The main script that allows to train, predict, and generate analysis outputs like heatmaps and activity labels.
  
- **`train.log`**: Stores logs generated during the training process, helpful for debugging and monitoring model performance.
  

## Running the Code

To train the model:

```
cd code/maincode python main.py --opt "train"`
```

To predict on a video:

```
python main.py --opt "predict" --path /path/to/video
```

For post-training analysis, such as generating heatmaps or activity labels:

1. **Generate Heatmap**:
  
  ```
  python main.py --opt "analysis" --path /path/to/video --method "heatmap"`
  ```
  
2. **Generate Activity Labels**:
  
  ```
  python main.py --opt "analysis" --path /path/to/video --method "activity_labels"`
  ```
  

## Dependencies

Ensure that all dependencies are installed:

```
pip install -r requirements.txt
```