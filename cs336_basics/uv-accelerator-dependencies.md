# Managing CPU and CUDA PyTorch Environments with uv

## Objective and constraints

The repository's staff-prescribed project dependencies include `torch~=2.11.0`. Accelerator selection should leave that default dependency list unchanged while providing reproducible local development environments with these properties:

- ordinary synchronization installs a CPU-only build;
- CUDA is an explicit opt-in on Windows and Linux;
- the CUDA environment includes the Triton package appropriate to its platform;
- the universal lock can describe both alternatives without downloading several gigabytes of wheels merely to inspect their metadata;
- CPU and CUDA selections cannot be requested together.

Dependency groups are a suitable boundary because they add development environments without redefining the package's runtime contract.

## Resolver behavior that shaped the configuration

Several details of Python packaging and uv resolution matter here.

First, PEP 440 treats a local build such as `2.11.0+cu130` as satisfying `==2.11.0` when the specifier itself has no local-version label. A CUDA wheel can therefore satisfy the staff requirement `torch~=2.11.0`. The CPU override uses `torch==2.11.0+cpu` on Windows and Linux, where PyTorch publishes an explicitly labeled CPU wheel. macOS uses strict equality, `torch===2.11.0`, because its ordinary wheel is the intended platform build.

Second, wheel compatibility covers operating system, architecture, Python ABI, and interpreter constraints. The resolver does not detect an NVIDIA GPU or inspect the installed driver. If a compatible CUDA wheel is visible and selected, uv can install it on a machine with no NVIDIA GPU; `torch.cuda.is_available()` then reports whether CUDA is usable at runtime. This is why CUDA remains an explicit group rather than an automatically selected hardware feature.

Third, a uv lock is universal by default. Direct wheel URLs for every supported platform and interpreter caused the lock operation to fetch complete wheels while resolving metadata. During the experiment, four Linux Torch wheels of roughly 400--506 MB each were downloaded, and the lock took about seven and a half minutes. Supplying PyTorch's flat package pages through `find-links` makes candidates and their metadata available to the resolver without encoding a matrix of direct wheel URLs.

A named PyTorch package index was also tested. Passing it as a general index affected resolution of unrelated packages because uv's index strategy gives an index ownership over package names it contains. For example, discovering `tqdm` on the PyTorch index prevented the resolver from freely choosing the PyPI candidate. Changing to an unsafe cross-index strategy would weaken dependency-confusion protection, so flat `find-links` pages were preferable for this narrowly scoped override.

Finally, local CUDA versions sort after the corresponding public release. If both CPU and CUDA candidates are visible without a fork, `2.11.0+cu130` can become the chosen universal/default solution. That behavior is internally consistent, but it makes a resource-intensive environment the default. Mutually exclusive CPU and CUDA groups let the lock record both branches while retaining CPU as the ordinary installation.

## Resulting configuration

The dependency groups declare exact accelerator builds and matching Triton packages:

```toml
[dependency-groups]
cuda = [
    "torch==2.11.0+cu130 ; sys_platform == 'win32' or sys_platform == 'linux'",
    "triton>=3.6,<3.7 ; sys_platform == 'linux'",
    "triton-windows>=3.6,<3.7 ; sys_platform == 'win32'",
]
cpu = [
    "torch==2.11.0+cpu ; sys_platform == 'win32' or sys_platform == 'linux'",
    "torch===2.11.0 ; sys_platform == 'darwin'",
]
```

The uv project configuration exposes the two PyTorch flat-package pages, makes the development and CPU groups default, and declares the accelerator groups incompatible:

```toml
[tool.uv]
package = true
python-preference = "managed"
default-groups = ["dev", "cpu"]
find-links = [
    "https://download.pytorch.org/whl/cpu/torch",
    "https://download.pytorch.org/whl/cu130/torch",
]
conflicts = [
    [
        { group = "cpu" },
        { group = "cuda" },
    ],
]
```

Without an explicit setting, uv behaves as though `default-groups = ["dev"]`; both `uv sync` and the more explicit `uv sync --dev` therefore include the development group. Defining `default-groups` replaces that list, so `dev` must appear beside `cpu` to preserve the development tools while making the CPU override automatic. The conflict declaration is also what permits uv to encode mutually exclusive dependency branches in one lock file. Since `cpu` is a default group, a CUDA synchronization must remove it explicitly:

```powershell
# Default development environment with CPU-only Torch
uv sync

# CUDA development environment
uv sync --group cuda --no-group cpu

# Validate or refresh the universal lock
uv lock --check
uv lock
```

Passing `--group cuda` alone is intentionally rejected because it would request both conflicting groups. The explicit command makes the environment transition visible at the command line. It does not need to warn about GPU availability: choosing the CUDA group is itself the programmer's accelerator selection.

## Platform-specific Triton support

PyTorch 2.11's compiler stack expects Triton 3.6. Linux uses the upstream `triton` package, while Windows uses the official `triton-windows` port. Both expose the `triton` Python module, so the platform markers select different distributions behind the same runtime import.

On the tested Windows machine, Visual Studio 2022 Build Tools were already installed but `cl.exe` was absent from an ordinary PowerShell environment. The x64 developer shell can be initialized with:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PYTHONUTF8 = "1"
```

UTF-8 mode was necessary on this localized Windows installation because PyTorch's compiler detection decoded `cl /help` using the system's GBK codec and otherwise raised `UnicodeDecodeError`. After synchronizing the CUDA group, the environment reported PyTorch `2.11.0+cu130` and Triton `3.6.0`; a small compiled CUDA function completed successfully. This establishes toolchain availability, although each benchmark still needs to control compilation and recompilation behavior carefully.

## Storage and memory cost

The explicit CPU default avoids a substantial cost on machines that cannot use CUDA. For CPython 3.12 on x86-64, the package pages reported these approximate downloads:

| Platform | CPU Torch | CUDA Torch and required CUDA packages | Additional download |
|---|---:|---:|---:|
| Windows | 114 MB | 1.92 GB for the Torch wheel | 1.80 GB |
| Linux | 190 MB | 2.71 GB including Torch, NVIDIA libraries, and Triton | 2.52 GB |

The local uv cache occupied about 0.398 GiB for the unpacked Windows CPU build and 2.632 GiB for the CUDA build, an increase of about 2.234 GiB. The synchronized CUDA virtual environment occupied about 2.652 GiB. A cache and virtual environment on different volumes may require copies instead of hard links, so both locations can bear most of that cost. In a simple local observation, importing CPU Torch used roughly 197 MiB of resident memory while the CUDA build used roughly 501 MiB; this approximately 300 MiB difference is environment-specific rather than a portable guarantee.

## Cache maintenance

uv provides cache-aware maintenance commands; manually removing cache internals is unnecessary:

```powershell
uv cache dir          # show the active cache directory
uv cache size         # report its size
uv cache prune        # remove entries no longer needed by current environments
uv cache clean torch  # remove cached artifacts for one package
uv cache clean        # clear the complete uv cache
```

`uv cache prune` is the conservative routine operation. Package-specific or complete cleaning is useful after deliberately switching large accelerator builds, with the expectation that later synchronization may download the removed artifacts again.

## References

- [uv dependency and group concepts](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [uv project configuration and conflicting groups](https://docs.astral.sh/uv/concepts/projects/config/)
- [uv's PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv cache behavior and maintenance](https://docs.astral.sh/uv/concepts/cache/)
- [PyTorch CPU wheel page](https://download.pytorch.org/whl/cpu/torch/)
- [PyTorch CUDA 13.0 wheel page](https://download.pytorch.org/whl/cu130/torch/)
- [Triton for Windows](https://github.com/triton-lang/triton-windows)
