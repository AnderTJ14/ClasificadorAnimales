const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const switchCameraButton = document.getElementById('switchCamera');
const result = document.getElementById('result');
const predictionSpan = document.getElementById('prediction');
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
    try {
        model = await tf.loadLayersModel('model/model.json');
        console.log('Modelo cargado correctamente');
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
    }
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
    try {
        const prediction = model.predict(imageTensor);
        prediction.print(); // Imprime los valores de predicción en la consola
        const predictedClass = prediction.argMax(1).dataSync()[0];
        console.log('Clase predicha:', predictedClass);
        const classes = ['Perro', 'Gato'];
        predictionSpan.textContent = classes[predictedClass];
    } catch (error) {
        console.error('Error en la predicción:', error);
    }
}

function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

video.addEventListener('loadedmetadata', () => {
    resizeCanvas();
});

async function init() {
    await setupCamera();
    await loadModel();
}

init();
