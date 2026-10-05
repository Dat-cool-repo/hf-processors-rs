// Run the wasm module in Node.js and compare every output with the native build
// (examples/native_reference.rs). Usage, from the repository root:
//
//   wasm-pack build bindings/wasm --target nodejs --out-dir pkg-node
//   cargo run --release -p hf-processors-wasm --example native_reference -- /tmp/ref golden IMAGES...
//   node bindings/wasm/tests/node_check.mjs bindings/wasm/pkg-node /tmp/ref golden IMAGES...
//
// Image cases must be bit-identical. Whisper is reported with its max abs difference (the
// math library behind f64 log10 / cos can differ between wasm and native in the last ulp).
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const [pkgDir, refDir, goldenDir, ...images] = process.argv.slice(2);
if (!images.length) {
  console.error("usage: node node_check.mjs PKG_DIR REF_DIR GOLDEN_DIR IMAGE...");
  process.exit(2);
}
const require = createRequire(import.meta.url);
const wasm = require(resolve(pkgDir, "hf_processors_wasm.js"));

// Keep in sync with examples/native_reference.rs.
const CONFIGS = [
  "openai_clip-vit-base-patch32.json",
  "google_siglip-so400m-patch14-384.json",
  "facebook_convnext-tiny-224.json",
  "Salesforce_blip-image-captioning-base.json",
  "Qwen_Qwen2-VL-2B-Instruct.json",
];

function compare(name, got) {
  const buf = readFileSync(join(refDir, name + ".f32"));
  const want = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (want.length !== got.length) return { identical: false, maxAbs: Infinity, n: got.length };
  let maxAbs = 0;
  let same = true;
  const gb = new Uint32Array(got.buffer, got.byteOffset, got.length);
  const wb = new Uint32Array(want.buffer.slice(want.byteOffset, want.byteOffset + want.byteLength));
  for (let i = 0; i < got.length; i++) {
    if (gb[i] !== wb[i]) {
      same = false;
      maxAbs = Math.max(maxAbs, Math.abs(got[i] - want[i]));
    }
  }
  return { identical: same, maxAbs, n: got.length };
}

let failures = 0;
let cases = 0;
let errors = 0;
for (const cfgName of CONFIGS) {
  const json = readFileSync(join(goldenDir, "configs", cfgName), "utf8");
  for (const backend of ["torchvision", "pil"]) {
    const p = new wasm.Preprocessor(json, backend);
    for (const image of images) {
      const name = `${cfgName.replace(/\.json$/, "")}__${backend}__${basename(image)}`;
      cases++;
      const errFile = join(refDir, name + ".err");
      if (existsSync(errFile)) {
        // The native build rejects this case (like transformers): the wasm build must too.
        const want = readFileSync(errFile, "utf8");
        let got = null;
        try {
          p.preprocessEncoded(readFileSync(image));
        } catch (e) {
          got = e.message;
        }
        if (got !== want) {
          failures++;
          console.log(`DIFF ${name}: expected error ${JSON.stringify(want)}, got ${JSON.stringify(got)}`);
        } else {
          errors++;
        }
        continue;
      }
      const t = p.preprocessEncoded(readFileSync(image));
      const r = compare(name, t.data);
      if (!r.identical) {
        failures++;
        console.log(`DIFF ${name}: max abs ${r.maxAbs} over ${r.n} values`);
      }
    }
  }
}
console.log(`images: ${cases - failures}/${cases} cases match native (${cases - failures - errors} bit-identical outputs, ${errors} identical errors)`);

const fe = new wasm.Preprocessor(readFileSync(join(goldenDir, "configs", "openai_whisper-tiny.json"), "utf8"));
const tone = new Float32Array(40000);
for (let i = 0; i < tone.length; i++) tone[i] = (((i * 7919) % 2000) - 1000) / 4096;
const w = fe.extractAudio(tone);
const r = compare("whisper", w.data);
console.log(`whisper: shape [${Array.from(w.shape)}], ${r.identical ? "bit-identical" : `max abs diff ${r.maxAbs}`} to native`);
if (!(r.maxAbs <= 1e-5)) failures++;
process.exit(failures ? 1 : 0);
