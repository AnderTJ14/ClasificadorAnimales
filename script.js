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
        // Convertir la imagen a escala de grises si el modelo espera imágenes en escala de grises
        const grayImageTensor = imageTensor.mean(3).expandDims(3); // Promediar los canales de color y añadir una dimensión de canal

        // Asegúrate de que el tensor de imagen tenga las dimensiones correctas
        console.log('Tensor de imagen (gris):', grayImageTensor.shape);

        // Realiza la predicción
        const prediction = model.predict(grayImageTensor);

        // Imprime los resultados de la predicción
        prediction.print(); // Esto te permitirá ver los valores de predicción en la consola

        // Obtén la clase con la mayor probabilidad
        const predictedClass = prediction.argMax(1).dataSync()[0];
        console.log('Clase predicha:', predictedClass);

        // Asegúrate de que la clase sea válida
        const classes = ['Perro', 'Gato'];
        predictionSpan.textContent = classes[predictedClass];
    } catch (error) {
        console.error('Error en la predicción:', error);
        predictionSpan.textContent = 'Error en la predicción';
    }
}

async function init() {
    await setupCamera();
    await loadModel();
}

init();
