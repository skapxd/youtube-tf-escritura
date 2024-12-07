import sharp from "sharp";
import fs from "fs";
import path from "path";

// Configuración
const inputPath: string = "./MNIST.png";
const outputDir: string = "./output-images/";
const imageSize: number = 28; // Tamaño original de cada imagen MNIST

/**
 * Crea el directorio de salida si no existe.
 */
function ensureOutputDirectory(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

async function bufferRow(imageBuffer: Buffer, index: number): Promise<Buffer> {
  return await sharp(imageBuffer)
    .extract({ top: index, left: 0, width: 784, height: 1 })
    .raw()
    .toBuffer();
}

async function saveImage(imageBuffer: Buffer, index: number) {
  const outputPath = path.join(outputDir, `image-${index}.png`);

  const row = await bufferRow(imageBuffer, index);

  await sharp(row, {
    raw: {
      width: imageSize,
      height: imageSize,
      channels: 3,
    },
  })
    .png()
    .toFile(outputPath);
}

/**
 * Procesa la imagen MNIST dividiéndola en imágenes individuales y ajustando su visibilidad.
 */
async function processMnistImage(): Promise<void> {
  try {
    // Asegurar que el directorio de salida existe
    ensureOutputDirectory(outputDir);

    // Cargar la imagen original
    const imageBuffer: Buffer = fs.readFileSync(inputPath);

    // Obtener dimensiones de la imagen
    const metadata = await sharp(imageBuffer).metadata();
    const width = metadata.width || 0;
    const height = metadata.height || 0;

    const images: Float32Array[] = [];
    for (let index = 0; index <= height; index++) {
      const array = new Float32Array(width);

      for (let i = 0; i < 784; i++) {
        // await saveImage(imageBuffer, index);

        const buffer = await bufferRow(imageBuffer, index);
        const item = buffer.readUInt8(index);

        array[i] = item / 255;
      }
      images.push(array);
    }
  } catch (error) {
    console.error("Error procesando imágenes:", error);
  }
}

// Ejecutar el procesamiento
processMnistImage();
