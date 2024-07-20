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
    let imageTensor = tf.browser.fromPixels(imageData).resizeNearestNeighbor([100, 100]).toFloat();

    // Convertir la imagen a escala de grises
    imageTensor = tf.image.rgbToGrayscale(imageTensor);

    // Asegurarse de que el tensor tenga 4 dimensiones: [batchSize, height, width, channels]
    imageTensor = imageTensor.expandDims(0); // [1, 100, 100, 1]

    predict(imageTensor);
});

switchCameraButton.addEventListener('click', () => {
    usingFrontCamera = !usingFrontCamera;
    setupCamera();
});

async function predict(imageTensor) {
    try {
        // Asegurarse de que el tensor tenga la forma correcta
        console.log('Forma del tensor:', imageTensor.shape);

        // Realizar la predicción
        const prediction = model.predict(imageTensor);

        // Asegurarse de que la predicción tenga la forma correcta
        const predictionArray = await prediction.array();
        console.log('Predicción:', predictionArray);

        // Obtener la clase con la mayor probabilidad
        const predictedClass = tf.tensor(predictionArray).argMax(1).dataSync()[0];
        console.log('Clase predicha:', predictedClass);

        // Asegúrate de que la clase sea válida
        const classes = ['Perro', 'Gato'];
        predictionSpan.textContent = `Predicción: ${classes[predictedClass]}`;
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
