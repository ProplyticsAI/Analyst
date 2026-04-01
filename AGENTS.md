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

### Current state
- Single file: `README.md`
- No programming language or framework has been chosen yet.
- No `package.json`, `requirements.txt`, `pyproject.toml`, `go.mod`, `Cargo.toml`, or any other dependency manifest exists.

### What future agents should do
Once application code is added, update this section with:
- How to install dependencies (update script).
- How to run lint, tests, build, and dev server.
- Any non-obvious environment caveats.
