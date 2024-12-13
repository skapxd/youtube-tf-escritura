import * as fs from "fs";
import * as path from "path";
import { Jimp } from "jimp";

const inputFile = "./png";
const outputDir = "./output-images";

// Crear carpeta de salida
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
} else {
  fs.rmdirSync(outputDir, { recursive: true });
  fs.mkdirSync(outputDir);
}

async function main() {
  const files = fs
    .readdirSync(inputFile)
    .filter((file) => file.endsWith(".png"));
  if (files.length === 0) {
    console.error("No se encontraron imágenes en el directorio especificado.");
    return;
  }

  for (const file of files) {
    const image = await Jimp.read(path.join(inputFile, file));

    console.log(`Procesando: ${file}`);

    // Redimensionar la imagen a 28x28
    image.resize({ w: 28, h: 28 });

    // Convertir la imagen a escala de grises
    image.greyscale();

    // Crear el archivo PGM manualmente
    const width = image.bitmap.width;
    const height = image.bitmap.height;
    const pixels = image.bitmap.data;

    // Crear cabecera del archivo PGM
    const header = `P5\n${width} ${height}\n255\n`;

    const pixelData = Buffer.alloc(width * height);

    let pixelIndex = 0;
    // Leer cada pixel en escala de grises
    for (let i = 0; i < pixels.length; i += 4) {
      const b = pixels[i];
      const g = pixels[i + 1];
      const r = pixels[i + 2];

      const grayValue = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      pixelData[pixelIndex++] = grayValue;
    }

    const headerBuffer = Buffer.from(header, "ascii");
    const pgmData = Buffer.concat([headerBuffer, pixelData]);

    // Guardar el archivo PGM
    const outputPath = path.join(outputDir, `${file.split(".")[0]}.pgm`);
    fs.writeFileSync(outputPath, pgmData, "utf8");
    console.log(`Archivo generado: ${outputPath}`);
  }

  console.log(`Archivos PGM generados exitosamente.`);
}

main();
