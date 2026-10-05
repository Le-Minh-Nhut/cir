import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = process.env.WORKBENCH_BACKEND_URL ?? "http://127.0.0.1";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": backendUrl } },
  preview: { proxy: { "/api": backendUrl } },
});
