
<p align="center">
  <img height="150" alt="Cotabby logo" src=".github/assets/readme/logo.png" />
</p>

<h1 align="center">Cotabby Neural</h1>

<p align="center"><em>Cotabby's local AI autocomplete, running silently on the Apple Neural Engine.</em></p>

<p align="center">
  <a href="https://github.com/FuJacob/cotabby"><img alt="Fork of FuJacob/cotabby" src="https://img.shields.io/badge/fork%20of-FuJacob%2Fcotabby-blue?logo=github" /></a>
  <a href="LICENSE"><img alt="License: AGPL v3" src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" /></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-lightgrey" />
  <img alt="Swift" src="https://img.shields.io/badge/Swift-F05138?logo=swift&amp;logoColor=white" />
</p>

> [!NOTE]
> This is a personal fork of **[Cotabby](https://github.com/FuJacob/cotabby)**, created by
> [@FuJacob](https://github.com/FuJacob) and maintained with [@jam-cai](https://github.com/jam-cai)
> and [@akramj13](https://github.com/akramj13). It adds one engine that runs on the Neural Engine and
> otherwise tracks upstream. For the official app, releases, and support, go to the
> [original project](https://github.com/FuJacob/cotabby) or [cotabby.app](https://cotabby.app).

---

## Why this fork

Cotabby's Open Source engine runs a small model on the GPU through llama.cpp. On my Mac, every
keystroke's burst of GPU work made the hardware whine audibly (coil whine). Apple Intelligence is
silent, but it is a chat model and handles half-typed words badly.

This fork runs the same kind of small **base** model on the **Apple Neural Engine** through Apple's
Core AI framework instead: silent, local, and about 110–125 ms per suggestion.

## What's different from upstream

- **A new Neural Engine engine.** It uses the same prompt and output path as the Open Source engine,
  but the model is a Core AI bundle running on the Neural Engine. The default model is
  Qwen3-0.6B-Base, picked after a Neural Engine-only quality comparison against Llama-3.2-1B.
- **macOS 27 and Apple Silicon only.** Core AI exists only on macOS 27, so the whole app now
  targets it, and it builds with Xcode 27.2.
- **No prebuilt releases and no auto-updates.** Build it from source. Upstream changes are merged
  in by hand.

Everything else is upstream Cotabby. [FORK.md](FORK.md) lists every change, how the model is made,
and how to merge new upstream code.

## What Cotabby does

Cotabby adds AI autocomplete to almost any text field on your Mac. As you type, a gray suggestion
appears inline next to your cursor. Press `Tab` to accept it a word at a time, or keep typing to
ignore it.

<div align="center">

|  |  |
|:---:|:---:|
| <img src="gifs/slack.gif" alt="Cotabby emoji autocomplete demo" width="400" height="225" /> | <img src="gifs/imessage.gif" alt="Cotabby autocomplete demo" width="400" height="225" /> |
| <img src="gifs/autocorrect.gif" alt="Cotabby autocorrect demo" width="400" height="225" /> | <img src="gifs/macros.gif" alt="Cotabby inline macros demo" width="400" height="225" /> |

</div>

- **Ghost-text autocomplete:** suggestions inline in almost any macOS text field; `Tab` accepts a
  word at a time.
- **Emoji autocomplete:** type `:rocket:` and accept it without leaving the field.
- **Inline macros:** type `/` for quick math, unit and currency conversion, dates, and random values.
- **One-key autocorrect:** fix a likely typo with a single keystroke.
- **Optional screen context:** turn visual context on or off separately from the other prediction
  settings.

## Engines

Pick one in Settings → Engine:

- **Neural Engine** *(this fork)*: a base model on the Apple Neural Engine. No GPU work, so no coil
  whine. The first load of a model, and the first load after each macOS update, takes a few minutes
  while Core AI prepares it. An hourglass in the menu bar shows that this is happening.
- **Apple Intelligence:** Apple's built-in on-device model.
- **Open Source:** a downloadable GGUF base model running on the GPU through
  [llama.cpp](https://github.com/ggerganov/llama.cpp).
- **OpenAI-compatible:** a completion or chat endpoint you configure (local Ollama, a LAN host, or
  a public HTTPS service).

## Build and run

Needs macOS 27, Apple Silicon, and Xcode 27.2. [CONTRIBUTING.md](CONTRIBUTING.md) covers the full
build and test workflow.

```bash
git clone https://github.com/eliakocher/cotabby-neural.git cotabby
cd cotabby
scripts/prepare_cotabby_workspace.sh
scripts/build_and_run.sh run Release
```

Then put a Neural Engine model bundle in `~/Library/Application Support/Cotabby Dev/NeuralEngine/`
and select **Neural Engine** in Settings. [FORK.md](FORK.md) has the command that exports
Qwen3-0.6B-Base into that format.

`build_and_run.sh` starts the app in debug mode, which logs every prompt to
`~/Library/Logs/Cotabby Dev/`. Quit it and reopen it from Finder for everyday use.

## Using Cotabby

When a gray suggestion appears:

- **`Tab`:** accept the next word. To accept whole phrases instead, change Settings → Acceptance
  Mode.
- **`` ` `` (backtick):** accept the entire suggestion.
- **`Esc`**, or keep typing: dismiss it.

Every shortcut can be changed under Settings → Shortcuts.

## Permissions and privacy

- **Accessibility:** read the text and cursor position in the field you're typing in, and insert
  what you accept.
- **Input Monitoring:** notice your typing and detect the accept keys.
- **Screen Recording** *(optional)*: visual context, and matching the ghost text's font to the
  app's text.

The Neural Engine, Apple Intelligence, and Open Source engines all generate on your Mac. There is no
analytics, telemetry, or crash reporting. Cotabby never suggests or inserts text in password and
other secure fields.

## Credits

- **[Cotabby](https://github.com/FuJacob/cotabby)** by [@FuJacob](https://github.com/FuJacob),
  [@jam-cai](https://github.com/jam-cai), and [@akramj13](https://github.com/akramj13). This fork
  is their app with one engine added. If you find it useful,
  [support the original project](https://ko-fi.com/cotabby).
- [apple/coreai-models](https://github.com/apple/coreai-models) (Core AI model export and runtime)
  and [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (the default Neural Engine model).
- Upstream's acknowledgments still apply: [llama.cpp](https://github.com/ggerganov/llama.cpp),
  [CotabbyInference](https://github.com/FuJacob/cotabbyinference),
  [Sparkle](https://github.com/sparkle-project/Sparkle), [swift-log](https://github.com/apple/swift-log),
  [LaunchAtLogin](https://github.com/sindresorhus/LaunchAtLogin-Modern),
  [GitHub gemoji](https://github.com/github/gemoji), and [SymSpell](https://github.com/wolfgarbe/SymSpell)
  with its frequency dictionaries (see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)).

## License

[GNU Affero General Public License v3.0](LICENSE), the same as upstream Cotabby. This is a modified
version, changed by [@eliakocher](https://github.com/eliakocher) starting in October 2026. The
changes are listed in [FORK.md](FORK.md), and the full source is this repository. Third-party
dependencies and model weights keep their own licenses.
