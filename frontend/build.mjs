// Bundle the React + TypeScript frontend with esbuild into frontend/dist (served by the backend).
// Usage (Node 18+): cd frontend && npm install && npm run build
import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const fe = dirname(fileURLToPath(import.meta.url));
const dist = join(fe, "dist");
rmSync(dist, { recursive: true, force: true });
mkdirSync(join(dist, "assets"), { recursive: true });
await build({
  entryPoints: { main: join(fe, "src", "main.tsx") },
  bundle: true, minify: true, sourcemap: false, format: "esm", target: ["es2020"],
  jsx: "automatic", outdir: join(dist, "assets"),
  loader: { ".woff2": "file", ".woff": "empty", ".ttf": "empty" }, // modern browsers only need woff2
  assetNames: "[name]-[hash]", publicPath: "/assets",
  define: { "process.env.NODE_ENV": '"production"' },
  logLevel: "info",
});
cpSync(join(fe, "index.html"), join(dist, "index.html"));
console.log("frontend built ->", dist);
