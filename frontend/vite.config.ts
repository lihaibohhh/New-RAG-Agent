import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      watch: {
        usePolling: env.CHOKIDAR_USEPOLLING === "true",
      },
      proxy: {
        "/api": {
          target:
            env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8010",
          changeOrigin: true,
        },
      },
    },
  };
});
