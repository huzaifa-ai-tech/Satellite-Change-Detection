import { fileURLToPath } from "url";
import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    react(),
    {
      name: "patch-leaflet-draw",
      transform(code, id) {
        if (id.endsWith("leaflet-draw/dist/leaflet.draw.js")) {
          return code.replace(
            "readableArea:function(e,i,o){var a,n,o=",
            "readableArea:function(e,i,o){var a,n,type,o=",
          );
        }
      },
    },
  ],

  resolve: {
    alias: [
      { find: /^leaflet-draw$/, replacement: path.resolve(__dirname, "src/leaflet-draw-shim.js") },
    ],
  },
});
