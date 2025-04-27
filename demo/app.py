from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import sys

# Add the path to the 'code' folder for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from component.predict import predict_video_class

# Initialize the Flask application
app = Flask(__name__)

# Configurations for file uploads and processed videos
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov', 'mkv'}

# Setting up upload and processed folder paths in the app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The name of the file to check.
    
    Returns:
        bool: True if the file has an allowed extension, otherwise False.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp4(input_path):
    """
    Converts a video file to MP4 format using ffmpeg.
    
    Args:
        input_path (str): The file path of the video to convert.
    
    Returns:
        str: The output path of the converted MP4 file, or None if conversion fails.
    """
    output_path = input_path.rsplit('.', 1)[0] + '.mp4'
    try:
        # Run the ffmpeg command to convert the video
        subprocess.run(['ffmpeg', '-y', '-i', input_path, output_path], check=True)
        return output_path
    except subprocess.CalledProcessError:
        return None

@app.route('/')
def index():
    """
    Renders the main index page.
    
    Returns:
        str: The HTML content of the index page.
    """
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    """
    Handles file upload, conversion, and video classification.
    
    If a file is uploaded, it is saved and processed. If necessary, it is converted to MP4,
    then classified using the 'predict_video_class' function.
    
    Returns:
        JSON response: The result of the prediction or error messages.
    """
    if request.method == 'POST':
        # Retrieve the uploaded video file
        file = request.files['video']
        
        # Validate the file type
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # If the file is in AVI format, convert it to MP4
            if file_ext == 'avi':
                converted = convert_to_mp4(filepath)
                if converted:
                    filepath = converted
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to convert AVI to MP4'})

            # Perform video classification
            result = predict_video_class(filepath)
            video_url = f"/uploads/{os.path.basename(filepath)}"
            return jsonify({'status': 'success', 'message': result, 'video_path': video_url})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    # Render the file upload page
    return render_template('upload.html')

@app.route('/record', methods=['GET', 'POST'])
def record_page():
    """
    Handles video recording and conversion. If a WebM file is uploaded, it is converted to MP4,
    then classified using the 'predict_video_class' function.
    
    Returns:
        JSON response: The result of the prediction or error messages.
    """
    if request.method == 'POST':
        # Retrieve the uploaded video file
        file = request.files['video']
        
        # Validate the file type
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # If the file is in WebM format, convert it to MP4
            if filename.rsplit('.', 1)[1].lower() == 'webm':
                converted_path = convert_to_mp4(filepath)
                if converted_path:
                    filepath = converted_path
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to convert WebM to MP4'})

            # Perform video classification
            result = predict_video_class(filepath)
            video_url = f"/uploads/{os.path.basename(filepath)}"
            return jsonify({'status': 'success', 'message': result, 'video_path': video_url})
        
        # Return an error if the file type is invalid
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    # Render the video recording page
    return render_template('record.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serves the uploaded video file.
    
    Args:
        filename (str): The name of the uploaded file to serve.
    
    Returns:
        Response: The video file served from the upload folder.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    """
    Starts the Flask web application.
    """
    app.run(debug=True)
