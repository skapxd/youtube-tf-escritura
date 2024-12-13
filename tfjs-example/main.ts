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

import tf from "@tensorflow/tfjs-node";
import argparse from "argparse";

import data from "./data";
import { getModel } from "./model";

async function run(epochs: string, batchSize: any) {
  await data.loadData();

  const model = getModel(data.LABEL_FLAT_SIZE);

  const { images: trainImages, labels: trainLabels } = data.getTrainData();
  model.summary();

  let epochBeginTime;
  let millisPerStep;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch = Math.ceil(
    numTrainExamplesPerEpoch / batchSize
  );
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
  });

  const { images: testImages, labels: testLabels } = data.getTestData();
  const evalOutput: any = model.evaluate(testImages, testLabels);

  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`
  );

  const modelSavePath = "./model";
  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: "TensorFlow.js-Node MNIST Example.",
  add_help: true,
});
parser.add_argument("--epochs", {
  type: "int",
  default: 20,
  help: "Number of epochs to train the model for.",
});
parser.add_argument("--batch_size", {
  type: "int",
  default: 128,
  help: "Batch size to be used during model training.",
});
const args = parser.parse_args();

run(args.epochs, args.batch_size);
