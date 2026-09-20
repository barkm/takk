// The API is a separate deployment (see ROADMAP-takk.md), so its origin is configurable. Empty in
// development and in `npm run preview`, where Vite proxies /api to the local server.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export const api = (path: string) => `${BASE}${path}`;
