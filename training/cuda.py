import os
from pathlib import Path
import site
import sys


def main():
    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    runtime = next((p / "cuda_runtime" for p in roots if (p / "cuda_runtime/include").is_dir()))
    libraries = [str(p) for root in roots for p in root.glob("*/lib") if p.is_dir()]
    cuda_root = Path(".cache/cuda").resolve()
    (cuda_root / "lib64").mkdir(parents=True, exist_ok=True)
    if not (cuda_root / "include").exists():
        (cuda_root / "include").symlink_to(runtime / "include", target_is_directory=True)
    for library in (runtime / "lib").glob("libcudart.so*"):
        target = cuda_root / "lib64" / library.name
        if not target.exists():
            target.symlink_to(library)
    unversioned = cuda_root / "lib64/libcudart.so"
    if not unversioned.exists():
        unversioned.symlink_to(next((runtime / "lib").glob("libcudart.so.*")))
    env = os.environ.copy()
    env["CUDA_PATH"] = str(cuda_root)
    env["CUDA_HOME"] = str(cuda_root)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(cuda_root / "lib64"), *libraries, env.get("LD_LIBRARY_PATH", "")]
    )
    os.execve(sys.executable, [sys.executable, "-u", *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
