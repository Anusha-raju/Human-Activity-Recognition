let mediaRecorder;
let recordedBlobs = [];
let stream;

const videoPreview = document.getElementById('recordPreview');
const playback = document.getElementById('playback');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

startBtn.onclick = async () => {
    recordedBlobs = [];

    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        videoPreview.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedBlobs.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedBlobs, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            playback.src = url;
            uploadRecordedVideo(blob);
        };

        mediaRecorder.start();
    } catch (err) {
        alert("Could not access camera.");
    }
};

stopBtn.onclick = () => {
    mediaRecorder.stop();
    stream.getTracks().forEach(track => track.stop());
};

function uploadRecordedVideo(blob) {
    const formData = new FormData();
    formData.append('video', blob, 'recorded_video.webm');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => alert(data.message));
}

// Uploading existing video
document.getElementById('uploadBtn').onclick = () => {
    const input = document.getElementById('videoUpload');
    const file = input.files[0];
    if (!file) return alert("Please select a video to upload.");

    document.getElementById('uploadedVideo').src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append('video', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('resultMsg').innerText = data.message;
    });
};
