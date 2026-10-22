# Cotabby, Neural Engine fork

[eliakocher/cotabby-neural](https://github.com/eliakocher/cotabby-neural), a personal fork of
[FuJacob/cotabby](https://github.com/FuJacob/cotabby). It adds one thing: a
**Neural Engine** engine that runs a small base model on the Apple Neural Engine (ANE) through
Apple's Core AI framework. Everything else is upstream Cotabby. The fork's work lives on the
`neural-engine` branch.

This file is the fork's own record. `README.md` is the fork's public front page: it is rewritten,
credits the original authors, and links back to upstream.

## Why

- **Coil whine.** The llama.cpp engine runs on the GPU (Metal), and every keystroke's burst of GPU
  work made the Mac whine audibly. The Neural Engine runs the same kind of model silently.
- **Base model, not a chat model.** Apple Intelligence is a chat model and handles half-typed words
  badly. The Neural Engine engine uses the same raw-continuation base-model path as llama.
- **Speed.** About 110–125 ms per suggestion (p50) in daily use.
- **Model choice: Qwen3-0.6B-Base**, picked over Llama-3.2-1B in a Neural Engine-only quality eval
  (Cotabby's phrase replay, screen split). Both scored the same on the next word (31.2% vs 31.6%).
  Qwen did better mid-word (33.2% vs 29.5%) and was about 2× faster (115 vs 215 ms p50).

**Trade-off:** Core AI exists only on macOS 27, so every target's deployment target is **macOS 27**
and building needs **Xcode 27.2**. That's why this fork can't go upstream as-is: upstream's CI
runs on macOS 15. Keep GitHub Actions disabled on the fork.

## What we added

New code (the engine itself):

| Path | What it does |
|---|---|
| `Cotabby/Services/Runtime/NeuralEngine/NeuralEngineRuntimeCore.swift` | Actor around Core AI's `StaticShapeEngine`, with a FIFO gate; tokenize, prefill, decode loop |
| `…/NeuralEngineRuntimeManager.swift` | UI-facing state, model picker, load/unload, the "preparing" flag |
| `…/NeuralEngineSuggestionEngine.swift` | `SuggestionGenerating` implementation; same prompt and output path as llama |
| `Cotabby/Support/Runtime/NeuralEngine/` | Pure logic: token classes from `tokenizer.json`, sampler chain and masks, KV prefix reuse, model folder scan |
| `Cotabby/Support/Runtime/TokenHealingVocabulary.swift` | Token-healing prefix matching |
| `Cotabby/UI/Settings/Panes/Engine/EngineAndModelPaneView+NeuralEngine.swift` | Settings: status, model picker, models folder |
| `CotabbyTests/Support/Runtime/NeuralEngine/`, `…/TokenHealingVocabularyTests.swift` | Unit tests for the pure logic |

Small edits to upstream files (wiring):

- `SuggestionEngineModels.swift`: the `.neuralEngine` engine kind and power profile.
- `SuggestionEngineRouter.swift`: routes to it, with no fallback to another engine.
- `CotabbyAppEnvironment.swift`, `AppDelegate.swift`: create the manager; load the model only while
  this engine is selected.
- `LlamaSuggestionEngine.swift`: `makeGenerationOptions` changed from private to internal, so both
  engines share one set of sampling options.
- A `case .neuralEngine` in the exhaustive switches: residency, selection, request factory,
  settings attention, settings model, and the Welcome/Context views.
- `MenuBarView.swift`, `MenuBarStatusLabelView.swift`, `HomePaneView.swift`, `CotabbyApp.swift`:
  show "Preparing…" during the one-time setup (an hourglass next to the menu bar icon).
- `project.yml`: deployment targets 27.0, plus the `CoreAIModels` package (`CoreAILM` product)
  pinned to a revision. `Config/Package.resolved`: its pins. `Cotabby.xcodeproj`: regenerated.
- `scripts/phrase_eval.py` and `PhrasePredictionEvalTests.swift`: `--neural-engine-bundle` runs
  the quality eval on the Neural Engine.
- `AGENTS.md`, `SOURCE_LAYOUT.md`, `CONTRIBUTING.md`: docs. `README.md`: rewritten as the fork's
  front page.

## Fixes on top of upstream

- **Ghost text in Zen/Firefox floated above the input box** on pages like Gemini. Gecko reports a
  field's character positions, and the field itself, shifted vertically (64 pt there). Only the
  bottom of its caret box sits on the real line, and that box's height varies. Cotabby trusted
  the shifted answer, because the field it checked against was shifted the same way.
  `GeckoCaretLinePolicy` (a pure rule, with tests) takes the line from the caret box's bottom and
  the height from the character answer whenever the two disagree. `BrowserAppDetector.isGeckoEngine`
  limits it to Firefox-engine apps; `AXTextGeometryResolver` applies it. This doesn't depend on
  macOS 27, so it could be offered upstream as a standalone PR.

## The model (not in git)

- **Bundle location:** `~/Library/Application Support/Cotabby Dev/NeuralEngine/qwen3_0_6b_base_qwen3_1_7b_6bit_static/`
  (496 MB). Back it up. The app uses whichever bundle folder sits there.
- **Recreating it:** clone `github.com/apple/coreai-models` at `9e12065`, then from its folder:
  `uv run coreai.llm.export Qwen/Qwen3-0.6B-Base --platform iOS --experimental
  --compute-precision float16 --compression-config models/qwen3/qwen3_1_7b_6bit.yaml
  --max-context-length 4096`.
  - Keep `--platform iOS`: that produces fixed-size shapes, which run on the ANE. The macOS export
    runs on the GPU.
  - Keep `--max-context-length 4096`: without it, the export builds a 32k-token context and a
    multi-GB KV cache.
  - Cost: about 3 min and 6 GB RAM. Don't run it alongside a build.
- **First load is slow.** The first load of a bundle, and the first load after every macOS update,
  takes about 7.5 min while Core AI prepares the model for the ANE. No suggestions appear until it
  finishes, and the menu bar shows an hourglass.
  - The prepared copy is cached in `~/Library/Caches/coreai-cache/<OS build>/com.jacobfu.tabby.dev/`.
  - The cache works only for the app that built it. Never copy it to another app or test runner;
    a copied cache loads fine, but every suggestion then fails.

## Day to day

- Build and run: `scripts/build_and_run.sh run Release`. This starts the app with `-cotabby-debug`,
  which writes every prompt to `~/Library/Logs/Cotabby Dev/llm-io.jsonl`. For normal use, quit it
  and reopen it from Finder, or log out and back in (it's a login item).
- Tests: see CONTRIBUTING.md (`xcodebuild test … -derivedDataPath build/DerivedData`), then
  `rm -rf build/DerivedData`.
- The dev app compiles Sparkle out, so it never auto-updates. Getting new upstream code means
  merging it (below).

## Merging upstream updates

Remotes:
- `origin` = `https://github.com/eliakocher/cotabby-neural.git`, the fork. Push here
  (`git push -u origin neural-engine` the first time).
- `upstream` = `https://github.com/FuJacob/cotabby.git`, fetch only. Its push URL is deliberately
  set to a dummy value, so a push can never land there by accident.

Setting this up on a fresh clone:

```bash
git clone https://github.com/eliakocher/cotabby-neural.git cotabby && cd cotabby
git remote add upstream https://github.com/FuJacob/cotabby.git
git remote set-url --push upstream DISABLED-push-to-your-fork-instead
```

Then, for every update:

```bash
git fetch upstream
git switch neural-engine
git merge upstream/main
```

Where conflicts usually happen, and how to resolve them:

- **`README.md`:** every upstream README edit conflicts, because ours is rewritten. Keep ours
  (`git checkout --ours README.md && git add README.md`). Then check whether upstream's change
  (a new feature, permission, or shortcut) belongs in our README too.
- **`Cotabby.xcodeproj/project.pbxproj`:** never hand-merge it. Resolve `project.yml` first, then
  run `xcodegen generate` and stage the project.
- **`project.yml`:** keep upstream's changes, plus our two: every `deploymentTarget` stays `27.0`
  (including any target upstream adds), and the `CoreAIModels` package and its dependency in the
  app template stay.
- **`Config/Package.resolved`:** take upstream's pins, but keep `coreai-models` and its transitive
  pins (eventsource, swift-asn1, swift-atomics, swift-collections, swift-crypto, swift-huggingface,
  swift-jinja, swift-nio, swift-system, swift-transformers, xgrammar, yyjson).
  **swift-collections must stay at 1.4.1.**
- **"Switch must be exhaustive" build errors:** upstream added a new `switch` over
  `SuggestionEngineKind` or `PowerProfile`. Add `case .neuralEngine`. It usually behaves like
  `.llamaOpenSource`: same prompt and request shape, but no model downloads and no fallback.
- **Upstream changed llama sampling or decoding** (sampler defaults, token healing, stop rules,
  KV-cache reuse in `LlamaRuntimeCore` or CotabbyInference): `NeuralEngineRuntimeCore` and
  `NeuralEngineSampler` copy that behavior by hand, so port the change. Shared pieces update
  automatically: the prompt renderer, `TokenHealingPlan`, `DecodeStopPolicy`, the confidence
  policy, the text normalizer, and `makeGenerationOptions`.

After merging: run the unit tests, `scripts/build_and_run.sh run Release`, and type a few
sentences with the Neural Engine selected.

## More detail

The full history (benchmarks, power measurements, every gotcha) is in
`.internal/neural-engine/CONTINUE-HERE.md`. It exists only on this Mac (`.internal/` is gitignored).
