import * as fs from "fs";
import * as path from "path";

const env = process.argv[2];
// Configuración
const labelFilePath = "./output-label/labels.txt"; // Archivo con las etiquetas
// const outputFile = "train-labels-idx1-ubyte"; // Archivo de salida IDX
const outputFile = env === "test"
? "t10k-labels-idx1-ubyte"
: "train-labels-idx1-ubyte"; // Archivo de salida IDX

// Función para crear el archivo IDX de etiquetas
async function createIDXLabelFile(labelPath: string, outputPath: string) {
  // Leer las etiquetas desde el archivo de texto
  const labelContent = fs.readFileSync(labelPath, "utf-8");
  const labels = labelContent
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "") // Eliminar líneas vacías
    .map((line) => parseInt(line, 10)); // Convertir a números

  const numLabels = labels.length;

  // Crear el encabezado IDX
  const magicNumber = 0x00000801; // Número mágico para etiquetas
  const headerBuffer = Buffer.alloc(8);
  headerBuffer.writeInt32BE(magicNumber, 0); // Número mágico
  headerBuffer.writeInt32BE(numLabels, 4); // Número de etiquetas

  // Crear el cuerpo del archivo IDX (las etiquetas)
  const labelBuffer = Buffer.alloc(numLabels);
  labels.forEach((label, index) => {
    labelBuffer.writeUInt8(label, index); // Cada etiqueta es un byte
  });

  // Combinar encabezado y cuerpo
  const outputBuffer = Buffer.concat([headerBuffer, labelBuffer]);

  // Escribir archivo IDX
  fs.writeFileSync(outputPath, outputBuffer);
  console.log(`Archivo IDX de etiquetas generado: ${outputPath}`);
}

// Ejecutar función
createIDXLabelFile(labelFilePath, outputFile).catch((err) =>
  console.error(err)
);
