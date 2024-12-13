import * as fs from "fs";
import * as path from "path";

const env = process.argv[2];
// Configuración

// Configuración
const inputFile =
  env === "test" ? "t10k-images-idx3-ubyte" : "train-images-idx3-ubyte"; // Archivo de salida IDX
// const inputFile = "generated-images.idx3-ubyte"; // Archivo IDX
const outputDir = "./output-images"; // Carpeta donde guardar las imágenes

// Crear carpeta de salida
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
} else {
  fs.rmdirSync(outputDir, { recursive: true });
  fs.mkdirSync(outputDir);
}

// Función principal para procesar el archivo
export async function processIDXFile(filePath: string) {
  const fileStream = fs.createReadStream(filePath);
  let headerBuffer = Buffer.alloc(16); // Para leer el encabezado
  let imageBuffer = Buffer.alloc(0); // Buffer acumulador
  let imageIndex = 0; // Índice de la imagen actual
  let imageSize = 0; // Tamaño de cada imagen

  const headerRead = await new Promise<void>((resolve, reject) => {
    fileStream.once("data", (chunk: Buffer) => {
      // Leer encabezado
      headerBuffer = chunk.slice(0, 16);
      imageBuffer = chunk.slice(16);

      resolve();
    });
  });

  // Extraer información del encabezado
  const magicNumber = headerBuffer.readInt32BE(0);
  const numImages = headerBuffer.readInt32BE(4);
  const numRows = headerBuffer.readInt32BE(8);
  const numCols = headerBuffer.readInt32BE(12);
  imageSize = numRows * numCols;

  console.log(`Magic Number: ${magicNumber}`);
  console.log(`Number of Images: ${numImages}`);
  console.log(`Imag e Dimensions: ${numRows}x${numCols}`);

  // Procesar imágenes completas
  while (imageBuffer.length >= imageSize) {
    const imageData = imageBuffer.slice(0, imageSize); // Extraer una imagen completa
    saveImage(imageData, numRows, numCols, imageIndex);
    imageBuffer = imageBuffer.slice(imageSize); // Eliminar la imagen procesada del buffer
    imageIndex++;
  }

  // Procesar imágenes del flujo
  fileStream.on("data", (chunk: Buffer) => {
    // Acumular datos
    imageBuffer = Buffer.concat([imageBuffer, chunk]);

    // Procesar imágenes completas
    while (imageBuffer.length >= imageSize) {
      const imageData = imageBuffer.slice(0, imageSize); // Extraer una imagen completa
      saveImage(imageData, numRows, numCols, imageIndex);
      imageBuffer = imageBuffer.slice(imageSize); // Eliminar la imagen procesada del buffer
      imageIndex++;
    }
  });

  fileStream.on("end", () => {
    console.log("Finished processing all images.");
  });
}

// Guardar cada imagen como un archivo .pgm
function saveImage(buffer: Buffer, rows: number, cols: number, index: number) {
  const filePath = path.join(outputDir, `image-${index}.pgm`);
  const pgmHeader = `P5\n${cols} ${rows}\n255\n`; // Encabezado PGM
  const pgmHeaderBuffer = Buffer.from(pgmHeader, "ascii");
  const pgmData = Buffer.concat([pgmHeaderBuffer, buffer]);

  fs.writeFileSync(filePath, pgmData);
  console.log(`Saved: ${filePath}`);
}

// Ejecutar la función
if (require.main === module) {
  processIDXFile(inputFile).catch((err) => console.error(err));
}
