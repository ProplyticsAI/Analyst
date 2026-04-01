# Agents

## Cursor Cloud specific instructions

This repository ("Analyst") is currently a blank scaffold with only a `README.md`. There are no application services, dependency manifests, build systems, tests, or lint configurations to run.

### Docker & NVIDIA Container Registry (nvcr.io)
- Docker is installed and configured with `fuse-overlayfs` storage driver (required for nested container environments).
- `iptables-legacy` is set as the default (required for Docker networking in this VM).
- The Docker daemon must be started with `sudo dockerd &>/tmp/dockerd.log &` before use.
- NVCR authentication uses the `Nvidia` secret (injected as env var). Login command:
  ```
  echo "$Nvidia" | sudo docker login nvcr.io --username '$oauthtoken' --password-stdin
  ```
- Use `sudo docker` for all Docker commands (or add user to docker group).

### NVIDIA NIM (Llama 3.2 3B Instruct)
- Image `nvcr.io/nim/meta/llama-3.2-3b-instruct:latest` (~20 GB) is pulled and cached.
- **GPU required**: The NIM container needs `--gpus all` with NVIDIA GPU hardware + `nvidia-container-toolkit`. The Cloud Agent VM has no GPU, so the container cannot serve inference here.
- NIM cache is at `~/.cache/nim` (mounted to `/opt/nim/.cache` inside the container).
- Run command (requires GPU host):
  ```
  export NGC_API_KEY="$Nvidia"
  export LOCAL_NIM_CACHE=~/.cache/nim
  sudo docker run -it --rm \
      --gpus all \
      --shm-size=16GB \
      -e NGC_API_KEY \
      -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
      -u $(id -u) \
      -p 8000:8000 \
      nvcr.io/nim/meta/llama-3.2-3b-instruct:latest
  ```
- When running, the API is at `http://localhost:8000` (OpenAI-compatible).

### Current state
- Single file: `README.md`
- No programming language or framework has been chosen yet.
- No `package.json`, `requirements.txt`, `pyproject.toml`, `go.mod`, `Cargo.toml`, or any other dependency manifest exists.

### What future agents should do
Once application code is added, update this section with:
- How to install dependencies (update script).
- How to run lint, tests, build, and dev server.
- Any non-obvious environment caveats.
