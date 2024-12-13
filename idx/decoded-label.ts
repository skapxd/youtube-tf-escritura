import * as fs from "fs";
import * as path from "path";

const env = process.argv[2];
// Configuración
const inputFile =
  env === "test" ? "t10k-labels-idx1-ubyte" : "train-labels-idx1-ubyte"; // Archivo de salida IDX

// const inputFile = "generated-images.idx3-ubyte"; // Archivo IDX
const outputDir = "./output-label"; // Carpeta donde guardar las imágenes

// Crear carpeta de salida
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
} else {
  fs.rmdirSync(outputDir, { recursive: true });
  fs.mkdirSync(outputDir);
}

// Función principal para procesar el archivo
async function processIDXFile(filePath: string) {
  const buffer = fs.readFileSync(filePath);

  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  ); // Convertir a ArrayBuffer
  const dataView = new DataView(arrayBuffer); // Ahora funciona correctamente

  const magicNumber = dataView.getInt32(0, false);
  if (magicNumber !== 2049) {
    throw new Error(`Formato IDX no compatible. Número mágico: ${magicNumber}`);
  }

  // Leer el número de etiquetas
  const numLabels = dataView.getInt32(4, false);

  // Leer las etiquetas
  const labels: number[] = [];
  for (let i = 8; i < 8 + numLabels; i++) {
    labels.push(dataView.getUint8(i));
  }

  saveLabel(labels);
}

// Guardar cada imagen como un archivo .pgm
function saveLabel(labels: number[]) {
  const filePath = path.join(outputDir, `labels.txt`);

  const data = labels.join("\n");
  fs.writeFileSync(filePath, data);
  console.log(`Saved: ${filePath}`);
}

// Ejecutar la función
processIDXFile(inputFile).catch((err) => console.error(err));
