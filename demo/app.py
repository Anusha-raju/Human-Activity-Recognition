from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import subprocess
import sys

# Add the path to the 'code' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from component.predict import predict_video_class

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp4(input_path):
    output_path = input_path.rsplit('.', 1)[0] + '.mp4'
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_path, output_path], check=True)
        return output_path
    except subprocess.CalledProcessError:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if file_ext == 'avi':
                converted = convert_to_mp4(filepath)
                if converted:
                    filepath = converted
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to convert AVI to MP4'})

            result = predict_video_class(filepath)
            video_url = f"/uploads/{os.path.basename(filepath)}"
            return jsonify({'status': 'success', 'message': result, 'video_path': video_url})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file type'})
    return render_template('upload.html')

@app.route('/record', methods=['GET', 'POST'])
def record_page():
    if request.method == 'POST':
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert webm to avi (or mpeg)
            if filename.rsplit('.', 1)[1].lower() == 'webm':
                converted_path = convert_to_mp4(filepath)
                if converted_path:
                    filepath = converted_path
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to convert WebM to AVI'})

            result = predict_video_class(filepath)
            video_url = f"/uploads/{os.path.basename(filepath)}"
            return jsonify({'status': 'success', 'message': result, 'video_path': video_url})
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    else:
        return render_template('record.html')



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
