{
  description = "Stanford CS336 Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Hermetic evaluation requires explicitly allowing unfree packages 
        # in case future CUDA inputs are shifted to Nix packages.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            uv
            ruff
            gh
            git
            curl
            wget
            zip
            unzip
            gnupg
            gnumake
            gcc
            
            # Dynamic linking dependencies for PyPI wheels (PyTorch, Triton)
            zlib
            glib
            stdenv.cc.cc.lib
            ncurses # Frequently required by Triton's LLVM backend
          ];

          PROJECT_NAME = "Stanford CS336";
          PYTHON_VERSION = "3.13";
          UV_INIT_BARE = "false";
          INSTALL_IPYKERNEL = "true";

          shellHook = ''
            export UV_CACHE_DIR="$PWD/.uv-cache"
            
            # 1. Interpreter Pinning:
            # Prevent `uv` from escaping the Nix sandbox and finding a system Python
            export UV_PYTHON="${pkgs.python313}/bin/python"

            # 2. Base Library Path:
            # Construct LD_LIBRARY_PATH so PyPI binaries can locate glibc and libstdc++
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
              stdenv.cc.cc.lib
              zlib
              glib
              ncurses
            ])}:$LD_LIBRARY_PATH"

            # 3. WSL2 GPU Passthrough:
            # WSL2 mounts the Windows NVIDIA drivers directly to this path. 
            # This is strictly required for `torch` and `cuda-bindings` to detect the GPU in WSL.
            if [ -d "/usr/lib/wsl/lib" ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi
          '';
        };
      }
    );
}