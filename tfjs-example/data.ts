/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from "@tensorflow/tfjs-node";
import assert from "assert";
import fs from "fs";
import https from "https";
import util from "util";
import zlib from "zlib";
import sharp from "sharp";

const readFile = util.promisify(fs.readFile);

// MNIST data constants:
const BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES_FILE = "train-images-idx3-ubyte";
const TRAIN_LABELS_FILE = "train-labels-idx1-ubyte";
const TEST_IMAGES_FILE = "t10k-images-idx3-ubyte";
const TEST_LABELS_FILE = "t10k-labels-idx1-ubyte";
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;

// Downloads a test file only once and returns the buffer for the file.
async function fetchOnceAndSaveToDiskWithBuffer(
  filename: string
): Promise<Buffer> {
  return new Promise((resolve) => {
    if (fs.existsSync(filename)) {
      resolve(readFile(filename));
      return;
    }
    const file = fs.createWriteStream(filename);
    const url = `${BASE_URL}${filename}.gz`;
    console.log(`  * Downloading from: ${url}`);
    https.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on("end", () => {
        resolve(readFile(filename));
      });
    });
  });
}

function loadHeaderValues(buffer: Buffer, headerLength: number): number[] {
  const headerValues: number[] = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }

  return headerValues;
}

async function loadImages(filename: string) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const dir = "./output-images/";
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images: Float32Array[] = [];
  let index = headerBytes;
  const imageCompress = Buffer.alloc(recordBytes * headerValues[1] * 4, 0);
  while (index < buffer.byteLength) {
    // 63_929 - 784_0016
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // Normalize the pixel values into the 0-1 interval, from
      // the original 0-255 interval.
      const item = buffer.readUInt8(index++);
      array[i] = item / 255;
      imageCompress.writeUInt8(item, index);
    }
    images.push(array);
  }

  assert.equal(images.length, headerValues[1]);
  return images;
}

async function loadLabels(filename: string) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels: Int32Array[] = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);
  return labels;
}

async function getLabelsCount(filename: string) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels: Int32Array[] = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);

  const cantidadDeEtiquetasUnicas = (() => {
    const arr = labels.map((e) => e.at(0));
    const total = new Set(arr).size;
    return total;
  })();

  return cantidadDeEtiquetasUnicas;
}

/** Helper class to handle loading training and test data. */
class MnistDataset {
  dataset: [
    Float32Array<ArrayBufferLike>[],
    Int32Array<ArrayBufferLike>[],
    Float32Array<ArrayBufferLike>[],
    Int32Array<ArrayBuffer>[]
  ];
  trainSize;
  testSize;
  trainBatchIndex;
  testBatchIndex;
  LABEL_FLAT_SIZE: number = 0;

  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  private async getTags() {
    const a = await getLabelsCount(TRAIN_LABELS_FILE);
    const b = await getLabelsCount(TEST_LABELS_FILE);

    if (a != b) throw new Error("Son diferentes longitudes");

    this.LABEL_FLAT_SIZE = a;
  }

  /** Loads training and test data. */
  async loadData() {
    await this.getTags();

    this.dataset = await Promise.all([
      loadImages(TRAIN_IMAGES_FILE),
      loadLabels(TRAIN_LABELS_FILE),
      loadImages(TEST_IMAGES_FILE),
      loadLabels(TEST_LABELS_FILE),
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData: boolean) {
    let imagesIndex;
    let labelsIndex;
    if (isTrainingData) {
      imagesIndex = 0;
      labelsIndex = 1;
    } else {
      imagesIndex = 2;
      labelsIndex = 3;
    }
    const size = this.dataset[imagesIndex].length;
    tf.util.assert(
      this.dataset[labelsIndex].length === size,
      () =>
        `Mismatch in the number of images (${size}) and ` +
        `the number of labels (${this.dataset[labelsIndex].length})`
    );

    // Only create one big array to hold batch of images.
    const imagesShape: [number, number, number, number] = [
      size,
      IMAGE_HEIGHT,
      IMAGE_WIDTH,
      1,
    ];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset);
      labels.set(this.dataset[labelsIndex][i], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf
        .oneHot(tf.tensor1d(labels, "int32"), this.LABEL_FLAT_SIZE)
        .toFloat(),
    };
  }
}

export default new MnistDataset();
