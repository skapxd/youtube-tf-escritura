import * as fs from "fs";
import * as path from "path";

const env = process.argv[2];
// Configuración
const imagesDir = "./output-images"; // Directorio con las imágenes en formato PGM
const outputFile = env === "test"
? "t10k-images-idx3-ubyte"
: "train-images-idx3-ubyte"; // Archivo de salida IDX

// Función principal
async function createIDXFile(imagesPath: string, outputPath: string) {
  const files = fs
    .readdirSync(imagesPath)
    .filter((file) => file.endsWith(".pgm"));
  if (files.length === 0) {
    console.error("No se encontraron imágenes en el directorio especificado.");
    return;
  }

  files.sort((a, b) => {
    const regex = /image-(\d+)\.pgm/;

    const _a = +a.match(regex)?.[1]!;
    const _b = +b.match(regex)?.[1]!;
    if (_a < _b) return -1;
    // if (b) return 1;
    return 0;
  });

  // Leer dimensiones de la primera imagen
  const firstImagePath = path.join(imagesPath, files[0]);
  const firstImageData = fs.readFileSync(firstImagePath);
  const { rows, cols } = parsePGMHeader(firstImageData);

  // Crear encabezado IDX
  const magicNumber = 0x00000803; // Para imágenes: 0x00000803 (big-endian)
  const numImages = files.length;
  const headerBuffer = Buffer.alloc(16);
  headerBuffer.writeInt32BE(magicNumber, 0); // Número mágico
  headerBuffer.writeInt32BE(numImages, 4); // Número de imágenes
  headerBuffer.writeInt32BE(rows, 8); // Número de filas
  headerBuffer.writeInt32BE(cols, 12); // Número de columnas

  // Crear contenido del archivo IDX
  const imageBuffers: Buffer[] = [headerBuffer];
  for (const file of files) {
    const filePath = path.join(imagesPath, file);
    const imageData = fs.readFileSync(filePath);
    const pixelData = extractPixelData(imageData);
    imageBuffers.push(pixelData);
    console.log(`Procesada: ${file}`);
  }

  // Escribir archivo IDX
  const outputBuffer = Buffer.concat(imageBuffers);
  fs.writeFileSync(outputPath, outputBuffer);
  console.log(`Archivo IDX generado: ${outputPath}`);
}

// Función para extraer dimensiones del encabezado PGM
function parsePGMHeader(buffer: Buffer): { rows: number; cols: number } {
  const header = buffer.toString("ascii", 0, buffer.indexOf("\n255\n") + 4);
  const lines = header
    .split("\n")
    .filter((line) => !line.startsWith("#") && line.trim() !== "");
  const [cols, rows] = lines[1].split(" ").map(Number);
  return { rows, cols };
}

// Función para extraer los datos de píxeles de una imagen PGM
function extractPixelData(buffer: Buffer): Buffer {
  const _ = buffer.slice(13);
  return _;
}

// Ejecutar función
createIDXFile(imagesDir, outputFile).catch((err) => console.error(err));
