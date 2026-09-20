import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

// In development and in `npm run preview` the API runs separately (`uv run takk`), and its routes are
// proxied so the page can call them at the same origin. A deployed build talks to VITE_API_BASE instead.
const proxy = { "/api": { target: "http://127.0.0.1:8002", changeOrigin: true } };

export default defineConfig({
  plugins: [sveltekit()],
  server: { proxy },
  preview: { proxy },
});
