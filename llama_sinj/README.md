# llama.cpp symbolic injection

This package contains the sinj patch for runtime residual-stream injection in
Qwen2 and Qwen2.5 models, plus the v2 Python tooling used to produce symbolic
files.

## Windows usage

Download and unzip the `llama-sinj-win-x64.zip` artifact from this repository's
GitHub Actions workflow. In PowerShell, start the server with a JSON sidecar:

```powershell
.\llama-server.exe -m <gguf> --jinja --sinj sidecar.json
```

Or load a `symbolic-file-v1` text file:

```powershell
.\llama-server.exe -m <gguf> --jinja --symbolic-file symbols.txt
```

Create a JSON sidecar with the extractor:

```powershell
.\llama-sinj-extract.exe -m <gguf> --text "love joy happy" `
  --contrast "hate anger sad" --layer 12 --point block --gain 4 `
  -o sidecar.json
```

The server provides `GET /sinj` to inspect active entries and `POST /sinj` to
replace or clear them. The same `llama-server` works with the llama-agent web
UI. This patch currently supports Qwen2 and Qwen2.5 only.

## Rebuild

Apply `sinj.patch` to a fresh `ggml-org/llama.cpp` b12057 checkout. Install
Visual Studio C++ build tools and the Vulkan SDK. The workflow uses MSVC with
CPU variants disabled:

```powershell
cmake -S . -B build `
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF `
  -DGGML_VULKAN=ON -DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release --target llama-server llama-cli `
  llama-completion llama-sinj-extract --parallel 2
```
