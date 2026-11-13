import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	agentRules: false, // dev server otherwise generates an untracked AGENTS.md
	// ort-web "bundle" embeds a 28 MB .wasm that would be emitted as a static asset (Workers limit: 25 MiB).
	// Use the extern-wasm build instead; the worker loads .mjs/.wasm from jsDelivr (see lib/inference/worker.ts).
	turbopack: {
		resolveAlias: {
			"onnxruntime-web/webgpu": "./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs",
		},
	},
};

export default nextConfig;

// Enable calling `getCloudflareContext()` in `next dev`.
// See https://opennext.js.org/cloudflare/bindings#local-access-to-bindings.
import { initOpenNextCloudflareForDev } from "@opennextjs/cloudflare";
initOpenNextCloudflareForDev();
