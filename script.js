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
        model = await tf.loadLayersModel('model/model.json');
        console.log('Modelo cargado correctamente');
        model.summary(); // Imprimir un resumen del modelo para verificar la arquitectura
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
    }
}

// Evento para capturar imagen y hacer predicción
captureButton.addEventListener('click', async () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    let imageTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([100, 100])
        .toFloat();
    
    // Normalizar la imagen
    imageTensor = imageTensor.div(tf.scalar(255.0));
    
    // Convertir a escala de grises
    imageTensor = tf.image.rgbToGrayscale(imageTensor);
    
    // Ajustar el tamaño y expandir dimensiones
    imageTensor = imageTensor.expandDims(0);

    // Verificar la forma del tensor
    console.log('Forma del tensor de entrada:', imageTensor.shape);
    
    // Realizar la predicción
    await predict(imageTensor);
});

// Cambiar entre cámara frontal y trasera
switchCameraButton.addEventListener('click', () => {
    usingFrontCamera = !usingFrontCamera;
    setupCamera();
});

// Realizar la predicción con el modelo
async function predict(imageTensor) {
    try {
        // Realizar la predicción
        const prediction = model.predict(imageTensor);
        const predictionArray = await prediction.array();
        const predictedClass = tf.tensor(predictionArray).argMax(1).dataSync()[0];
        const classes = ['Perro', 'Gato'];
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
