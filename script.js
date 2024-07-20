const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const switchCameraButton = document.getElementById('switchCamera');
const result = document.getElementById('result');
const context = canvas.getContext('2d');
let model;
let currentStream;
let usingFrontCamera = true;

async function setupCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: usingFrontCamera ? 'user' : 'environment'
        },
    });
    video.srcObject = stream;
    currentStream = stream;
}

async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
}

captureButton.addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const imageTensor = tf.browser.fromPixels(imageData).resizeNearestNeighbor([100, 100]).toFloat().expandDims();
    predict(imageTensor);
});

switchCameraButton.addEventListener('click', () => {
    usingFrontCamera = !usingFrontCamera;
    setupCamera();
});

async function predict(imageTensor) {
    const prediction = model.predict(imageTensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];
    const classes = ['Perro', 'Gato'];
    result.textContent = `Resultado: ${classes[predictedClass]}`;
}

async function init() {
    await setupCamera();
    await loadModel();
}

init();


