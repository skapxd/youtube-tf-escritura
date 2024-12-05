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

    const outputPath = (index: number) =>
      path.join(outputDir, `image-${index}.png`);

    for (let index = 0; index <= height; index++) {
      await sharp(imageBuffer)
        .extract({ top: index, left: 0, width, height: 1 })
        .raw()
        .toBuffer()
        .then(async (data) => {
          await sharp(data, {
            raw: {
              width: imageSize,
              height: imageSize,
              channels: 3,
            },
          })
            .png()
            .toFile(outputPath(index));
        });
    }
  } catch (error) {
    console.error("Error procesando imágenes:", error);
  }
}

// Ejecutar el procesamiento
processMnistImage();
