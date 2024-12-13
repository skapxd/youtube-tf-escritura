import * as tf from "@tensorflow/tfjs-node";
import sharp from "sharp";
import fs from "fs";
import { randomUUID } from "crypto";
import { join } from "path";

const PIXEL_SIZE_RGBA = 4;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

const assetsDir = join(__dirname, "assets");
if (!fs.existsSync(assetsDir)) {
  fs.mkdirSync(assetsDir, { recursive: true });
}
/**
 * Convert one-hot encoded labels to class indices.
 * @param {Uint8Array} oneHotLabels - One-hot encoded labels array.
 * @param {number} numClasses - Number of classes in the dataset.
 * @returns {Int32Array} - Array of class indices.
 */
function oneHotToIndices(oneHotLabels, numClasses) {
  const numLabels = oneHotLabels.length / numClasses;
  const indices = new Int32Array(numLabels);

  for (let i = 0; i < numLabels; i++) {
    const offset = i * numClasses;
    for (let j = 0; j < numClasses; j++) {
      if (oneHotLabels[offset + j] === 1) {
        indices[i] = j;
        break;
      }
    }
  }

  return indices;
}
/**
 * Convert class indices to one-hot encoded labels.
 * @param {Int32Array | Array} indices - Array of class indices.
 * @param {number} numClasses - Total number of classes.
 * @returns {Uint8Array} - One-hot encoded labels array.
 */
function indicesToOneHot(indices, numClasses) {
  const numLabels = indices.length;
  const oneHotLabels = new Uint8Array(numLabels * numClasses);

  for (let i = 0; i < numLabels; i++) {
    const classIndex = indices[i];
    oneHotLabels[i * numClasses + classIndex] = 1;
  }

  return oneHotLabels;
}

async function bufferRow(imageBuffer: Buffer, index: number): Promise<Buffer> {
  return await sharp(imageBuffer)
    .extract({ top: index, left: 0, width: 784, height: 1 })
    .raw()
    .toBuffer();
}

// Función para guardar las imágenes
class MnistData {
  private shuffledTrainIndex = 0;
  private shuffledTestIndex = 0;
  private datasetImages!: Float32Array[];
  private datasetLabels!: Uint8Array;
  private trainIndices!: Uint32Array;
  private testIndices!: Uint32Array;
  private trainImages!: Float32Array;
  private testImages!: Float32Array;
  private trainLabels!: Uint8Array;
  private testLabels!: Uint8Array;

  async load() {
    const imageBuffer = await this.fetchBuffer(MNIST_IMAGES_SPRITE_PATH);
    const labelBuffer = await this.fetchBuffer(MNIST_LABELS_PATH);

    const datasetBytesBuffer = new ArrayBuffer(
      NUM_DATASET_ELEMENTS * IMAGE_SIZE * PIXEL_SIZE_RGBA
    );

    const sprite = sharp(imageBuffer)
      .raw()
      .toBuffer({ resolveWithObject: true });
    const { data, info } = await sprite;

    const images: Float32Array[] = [];

    for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
      const array = new Float32Array(784);

      for (let j = 0; j < IMAGE_SIZE; j++) {
        const buffer = await bufferRow(imageBuffer, i);
        const item = buffer.readUInt8(i);
        array[i] = item / 255;
      }
      images.push(array);
    }

    this.datasetImages = images;
    this.datasetLabels = new Uint8Array(labelBuffer);

    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    );

    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);

    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );

    this.testLabels = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
  }

  private async fetchBuffer(url: string): Promise<Buffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return Buffer.from(await response.arrayBuffer());
  }

  nextBatch(
    batchSize: number,
    data: [Float32Array, Uint8Array],
    index: () => number
  ) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();
      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      );
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      );
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
    return { xs, labels };
  }

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }
}

function getModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: [3, 3],
      activation: "relu",
      inputShape: [28, 28, 1],
    })
  );

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: [3, 3],
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: 100,
      activation: "softmax",
    })
  );
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "softmax",
    })
  );
  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

async function trainModel() {
  const data = new MnistData();
  await data.load();
  const model = getModel();

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 55000;
  const TEST_DATA_SIZE = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
  });
}

trainModel().catch(console.error);
