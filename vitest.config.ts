
import { resolve } from "node:path";

import { configDefaults, defineConfig } from "vitest/config";
import { duration } from "./utils/duration";

const testResultDir = resolve(__dirname, "test-reporter");
const alias = {
  "#/": resolve(__dirname),
};

export default defineConfig({
  test: {
    hookTimeout: duration({ minute: 1 }),
    testTimeout: duration({ minute: 1 }),
    coverage: {
      enabled: true,
      provider: "v8",
      reporter: ["text", "html", "clover", "lcov", "cobertura"],
      reportsDirectory: resolve(testResultDir, "coverage"),
    },
    outputFile: {
      html: resolve(testResultDir, "index.html"),
      junit: resolve(testResultDir, "junit-report.xml"),
    },
    exclude: [...configDefaults.exclude, "e2e/**/**"],
    reporters: ["default", "html", ["junit", { suiteName: "UI tests" }]],
    globals: true,
    root: "./",
  },
  resolve: {
    alias,
  },
});
