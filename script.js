const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const switchCameraButton = document.getElementById('switchCamera');
const predictionSpan = document.getElementById('prediction');
const context = canvas.getContext('2d');
let model;
let currentStream;
let usingFrontCamera = true;

// Configurar la cámara
async function setupCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: usingFrontCamera ? 'user' : 'environment' }
    });
    video.srcObject = stream;
    currentStream = stream;
}

// Cargar el modelo de TensorFlow.js
async function loadModel() {
    try {
        model = await tf.loadLayersModel('https://raw.githubusercontent.com/AnderTJ14/ClasificadorAnimales/main/model/model.json');
        console.log('Modelo cargado correctamente');
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
    }
}

// Evento para capturar imagen y hacer predicción
captureButton.addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    let imageTensor = tf.browser.fromPixels(canvas).resizeNearestNeighbor([100, 100]).toFloat();
    imageTensor = tf.image.rgbToGrayscale(imageTensor);
    imageTensor = imageTensor.expandDims(0); // Asegura la forma [1, 100, 100, 1]
    predict(imageTensor);
});

// Cambiar entre cámara frontal y trasera
switchCameraButton.addEventListener('click', () => {
    usingFrontCamera = !usingFrontCamera;
    setupCamera();
});

// Realizar la predicción con el modelo
async function predict(imageTensor) {
    try {
        const prediction = model.predict(imageTensor);
        const predictionArray = await prediction.array();
        const predictedClass = tf.tensor(predictionArray).argMax(1).dataSync()[0];
        const classes = ['Perro', 'Gato']; // Asegúrate de actualizar con tus clases
        predictionSpan.textContent = `Predicción: ${classes[predictedClass]}`;
    } catch (error) {
        console.error('Error en la predicción:', error);
        predictionSpan.textContent = 'Error en la predicción';
    }
}

// Inicializar la aplicación
async function init() {
    await setupCamera();
    await loadModel();
}

init();
