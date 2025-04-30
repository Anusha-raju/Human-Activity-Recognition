# Demo Directory Overview
This folder contains the Flask web application for the Human Activity Recognition (HAR) project. The app allows users to upload video files and receive real-time activity predictions from the trained model. It provides an interactive way to test the HAR system.

Project Demo: [Human Activity Recognition using LRCN](https://youtu.be/lslaPkM5nkA)

## Folder Structure

```plaintext
demo/
├── static/                   # Static files like CSS, JS, images for Flask app
├── templates/                # HTML templates for Flask app
├── app.py                    # Flask server script for uploading and predicting activities
```

### `static/`

Contains static assets for the web application:

- **CSS**: Stylesheets for the frontend.
  
- **JavaScript**: Frontend logic and interactivity.
  
- **Images**: Any images needed for the web app.
  

### `templates/`

Contains HTML templates for rendering the frontend of the web application.

### `app.py`

The main Flask application script that:

- Serves the web interface.
  
- Handles video uploads and recording of a video
  
- Interacts with the model to make activity predictions.
  
- Displays results back to the user.
  

## Running the Demo

To run the Flask application:

```
sudo apt update
sudo apt install ffmpeg
```

1. Navigate to the `demo/` folder:
  
  ```
  cd demo/
  ```
  
2. Start the Flask server:
  
  ```
  python3 app.py
  ```
  

The application will be hosted locally on port 5000.

You can access it via: http://127.0.0.1:5000/

### Upload a Video

- Use the web interface to upload a video.
  
- The model will process the video and predict the activity in the video.
  
- The prediction result will be displayed on the web page.
  

## Dependencies

Ensure that all dependencies are installed:

```
pip install -r requirements.txt`
```