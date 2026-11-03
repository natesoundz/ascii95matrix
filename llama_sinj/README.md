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

Apply `sinj.patch` to a fresh `ggml-org/llama.cpp` b12057 checkout. Then install
the Vulkan SDK, Ninja, and LLVM/Clang, and configure/build with the same Windows
settings as `.github/workflows/llama-sinj-windows.yml`:

```powershell
cmake -S . -B build -G "Ninja Multi-Config" `
  -D CMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake `
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON `
  -DGGML_VULKAN=ON -DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release --target llama-server llama-cli `
  llama-completion llama-sinj-extract
```
