import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";
import { generateRoutes } from "./scripts/generate-routes.mjs";
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, fileURLToPath(new URL(".", import.meta.url)), "VITE_");
  const apiTarget =
    process.env["VITE_API_PROXY_TARGET"] ?? env["VITE_API_PROXY_TARGET"] ?? "http://127.0.0.1:8000";
  const devPort = Number(process.env["VITE_DEV_PORT"] ?? env["VITE_DEV_PORT"] ?? 5173);
  return {
    plugins: [
      {
        name: "satmod-file-routes",
        async buildStart() {
          await generateRoutes();
        },
        configureServer(server) {
          server.watcher.on("add", (path) => {
            if (path.includes("/src/routes/")) void generateRoutes();
          });
        },
      },
      react(),
      tailwindcss(),
    ],
    resolve: {
      dedupe: ["react", "react-dom", "@tanstack/react-query"],
      alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
    },
    // The local Plotly distribution is UMD; linked packages need explicit optimization.
    optimizeDeps: { include: ["plotly.js-dist-min"] },
    server: {
      host: "127.0.0.1",
      port: devPort,
      strictPort: true,
      proxy: {
        "/api": apiTarget,
        "/docs": apiTarget,
        "/openapi.json": apiTarget,
      },
    },
    preview: { host: "127.0.0.1", port: 5173, strictPort: true },
    build: {
      chunkSizeWarningLimit: 5000,
      commonjsOptions: { include: [/node_modules/, /vendor[\\/]+plotly\.js-dist-min/] },
    },
  };
});
