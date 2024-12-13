import * as tf from "@tensorflow/tfjs-node";
import { readFileSync } from "fs";

// Cargar el modelo
async function loadModel(modelPath: string) {
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
}

// Convertir la imagen a un tensor
function imageToTensor(imagePath: string): tf.Tensor {
  const buffer = readFileSync(imagePath);
  const arr = Array.from(buffer.slice(13)).map((value) => value / 255);
  const tensor = tf.tensor(arr, [1, 28, 28, 1]);
  return tensor;
}

// Hacer una predicción
async function predict(modelPath: string, imagePath: string) {
  const model = await loadModel(modelPath);
  const inputData = imageToTensor(imagePath);
  const prediction = model.predict(inputData) as tf.Tensor;
  prediction.print();
}

// Ejemplo de uso
const modelPath = "./model/model.json";
const imagePath = "./1-0.pgm"; // Ruta a la imagen
predict(modelPath, imagePath);
