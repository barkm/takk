import adapter from "@sveltejs/adapter-static";

// A static site: no server runs the frontend, so it can be hosted anywhere, apart from the API
// (see the deployment decision in ROADMAP-takk.md).
export default {
  kit: {
    adapter: adapter({ fallback: "index.html" }),
  },
};
