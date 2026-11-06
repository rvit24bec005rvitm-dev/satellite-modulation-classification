# SatModClass frontend

The integrated Lovable interface runs with Vite and file-based TanStack Router. Start the Python API on port 8000, then run `npm install` and `npm run dev` here. The local Vite proxy forwards `/api` to FastAPI.

See the [repository README](../README.md) for prerequisites, the combined launcher, model artifacts, safe upload formats, production hosting, tests, and known validation limits.

Commands: `npm run typecheck`, `npm run lint`, `npm test`, `npm run test:integration`, `npm run build`, `npm run test:e2e`.

Edit `src/routes/*.tsx` to change workspaces. `scripts/generate-routes.mjs` generates the flat route tree before dev/build/test/typecheck. The shared mission-control shell preserves the original UI primitives and all eight workspaces. Overview loads a real Three.js / React Three Fiber globe and a local Leaflet map; all orbit data is explicitly simulated. Full captures live in shared context; pages use React Query and the typed `src/lib/satmod-api.ts` client.
