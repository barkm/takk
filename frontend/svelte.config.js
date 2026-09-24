import adapter from "@sveltejs/adapter-static";

// A static site: no server runs the frontend, so it can be hosted anywhere, apart from the API
// (see the deployment decision in ROADMAP-takk.md). GitHub Pages serves a project site under
// /<repo>/, so the build takes that prefix from BASE_PATH (empty in development), and every link
// goes through `base` from $app/paths. The fallback is 404.html because that is the page Pages
// serves for a path it has no file for, which is how a single-page app is routed there.
export default {
  kit: {
    adapter: adapter({ fallback: "404.html" }),
    paths: { base: process.env.BASE_PATH ?? "" },
  },
};
