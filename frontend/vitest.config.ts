import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "jsdom", globals: true },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "plotly.js-dist-min": path.resolve(__dirname, "src/test/plotly-stub.ts"),
    },
  },
  esbuild: { jsx: "automatic" },
});
