# Human Activity Recognition (HAR) with LRCN

## Abstract

This project builds a **Human Activity Recognition** system using the **Long-term Recurrent Convolutional Network (LRCN)** architecture to accurately classify activities from video data. It features a user-friendly **Flask web application** that allows users to upload videos and instantly receive activity predictions, creating an interactive and accessible experience.

Project Demo:

Project Github:

---

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
  

```
git clone https://github.com/Anusha-raju/Human-Activity-Recognition.git
```

2. Create a Virtual Environment:
  

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
  

```
pip install -r requirements.txt
```

---

## Dataset

The **UCF50** dataset is a action recognition dataset consisting of realistic videos from YouTube, covering **50 different action categories**. It extends the earlier **UCF11** dataset and focuses on providing natural, unstaged video clips with large variations in camera motion, background, viewpoint, and lighting. Each action category is organized into **25 groups**, with each group sharing similarities like background, subjects, or camera angles. The dataset includes diverse activities such as Baseball Pitch, Jump Rope, Skiing, Playing Guitar, Walking with a Dog, and many more.

To download the dataset run the following command on your linux environment:

```
wget https://dlproject2025.s3.us-east-1.amazonaws.com/UCF50.zip
```

To unzip the datafile:

```
unzip UCF50.zip
```

This creats a folder `Data\UCF50` which has subfolders for each action.

---

## Model Architecture

This project uses an **LRCN** (CNN + LSTM) model:

- CNN: Extracts spatial features from video frames.
- LSTM: Captures temporal dependencies across frame sequences.
- Final FC layer: Classifies into one of the predefined activity classes.

Final Model Trained:

---

## Usage

The project provides scripts for training, predicting and analysing the model. Here are some examples of how to use the scripts:

1. To train and evaluate the model:
  
  ```
  cd code/maincode
  python main.py --opt "train"
  ```
  

2. To Predict on a video clip:
  
  ```
  python main.py --opt "predict" --path /path/to/video
  ```
  
3. Post-Training Analysis:
  
  1. Generate Heatmap:
    
  
  ```
  python main.py --opt "analysis" --path /path/to/video --method "heatmap"
  ```
  
  2. Generate per-frame activity labels:
    
  
  ```
  python main.py --opt "analysis" --path /path/to/video --method "activity_labels"
  ```
  

---

## Demo:

To run the flask application

```
cd demo/
python3 app.py 
```

The app will be hosted in 5000 port

Example: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## Files Overview

The project has 5 folders `code`, `demo`, `presentation`, `project_report`, `proposal`.

code: It has all the code files for training, prediction and analysis.

demo: It contains the scripts to host the **Flask** application.

presentation: It has the presentation slides of the project.

project_report: It has the project report.

proposal: It has the topic proposal file.

#### Folder Structure

```plaintext
Human-Activity-Recognition/
├── code/                         # Code files
│   ├── component/                # Core modules for training and prediction logic
|   |   ├── utils/                # Preprocessing and common code modules
│   │   |    ├── dataset.py       # Code to load and process the data.
│   │   |    ├── utility.py       # Common Functions
│   │   |    ├── visualization.py # Code to generate Plot metrics
│   │   ├── model.py              # Model Definition
│   │   ├── post_training.py      # Code to perform post training analysis
│   │   ├── predict.py            # Code to handle model predictions
│   │   ├── train.py              # Code to handle model training
│   ├── maincode/                 # Main execution scripts and model outputs
│   │   ├── models/               # Saved trained models
│   │   ├── outputs/              # Folder for storing evaluation results
│   │   ├── main.py               # Entry script to run training, evaluation, and heatmap generation
│   │   ├── train.log             # Training logs
├── Data/                         # Folder for dataset (e.g., UCF50 videos)
├── demo/                         # Web app (Flask) related files
│   ├── static/                   # Static files like CSS, JS, images for Flask app
│   ├── templates/                # HTML templates for Flask app
│   ├── app.py                    # Flask server script to upload and predict activities
├── presentation/                 # Presentation materials (slides)
├── project_report/               # Final project report files
├── proposal/                     # Project proposal documents
├── .env                          # Environment variables (e.g., dropout rate, paths)
├── requirements.txt              # List of Python dependencies
├── README.md                     # Project documentation
├── venv/                         # Virtual environment (not to be edited manually)
```

Please visit individual directories for an extensive documentation.

## Contributors:

[Anusha Umashankar](https://github.com/Anusha-raju)