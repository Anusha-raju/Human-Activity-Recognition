import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile

def convert_video(uploaded_file):
    # Create a temporary directory to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        # Save the uploaded file to the temporary directory
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Now, use the temporary file path with VideoFileClip
    video = VideoFileClip(tmp_file_path)
    
    # Convert video and save it to a new file
    output_file = "video.mp4"  # Specify the desired output path
    video.write_videofile(output_file, codec='libx264')

    # Clean up the temporary file
    # os.remove(tmp_file_path)

    return output_file

def main():
    st.title("Video Format Converter and Player")
    
    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'webm'])
    
    if video_file is not None:
        # Show the uploaded video details
        st.write(f"Video uploaded: {video_file.name}")
        
        # Convert the video to MP4 format
        converted_video_path = convert_video(video_file)
        
        # Play the converted video
        st.video(converted_video_path, format="video/mp4")
        st.write("Playing the converted video...")

if __name__ == "__main__":
    main()

