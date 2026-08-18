import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

from semantic_transformer.api import (
    transform_directory,
)
from semantic_transformer.core.transformation_name import (
    TransformationName,
)


def venv_python_bin(venv_path: Path) -> str:
    if os.name == "nt":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")


def venv_pip_bin(venv_path: Path) -> str:
    if os.name == "nt":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")


def run_command(cmd, cwd=None, env=None, capture=False):
    print(f">>> RUN: {cmd} (in {cwd})")
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
    )
    output = ""
    if capture:
        for line in process.stdout:
            output += line
            print(line, end="")
    process.wait()
    return process.returncode, output


def make_venv(venv_path: Path):
    venv_path = Path(venv_path)
    if venv_path.exists():
        print(f"Removing existing venv at {venv_path}")
        shutil.rmtree(venv_path)
    cmd = f"{sys.executable} -m venv {str(venv_path)}"
    rc, _ = run_command(cmd)
    if rc != 0:
        raise RuntimeError(f"Failed to create venv at {venv_path}")
    print(f"Created venv at {venv_path}")
    return venv_path


def git_clone(repo_url: str, dest: Path):
    if dest.exists():
        print(f"Destination {dest} already exists. Removing.")
        shutil.rmtree(dest)
    cmd = f"git clone {repo_url} {dest}"
    rc, _ = run_command(cmd)
    if rc != 0:
        raise RuntimeError("git clone failed")
    print(f"Cloned {repo_url} into {dest}")


def run_tests_with_coverage(repo_dir: Path, report_dir: Path, log_prefix: str, venv_python: str):
    run_command(
        f"cd {repo_dir} && {venv_python} -m pytest --cov > {report_dir}/{log_prefix}_pytest_results.log 2>&1"
    )


def install_deps(repo_dir: Path, venv_python: str):
    run_command(
        f"cd {repo_dir} && {venv_python} -m pip install . && {venv_python} -m pip install simplejson hypothesis pytest-cov"
    )


def delete_temp_files(work_dir: Path):
    if work_dir.exists():
        print(f"Cleaning up work dir {work_dir}")
        shutil.rmtree(work_dir)


def apply_transform_to_repo(
    repo_dir: Union[str, Path],
    transform_func: Callable[..., Optional[str]],
    exclude_dirs: Optional[Set[str]] = None,
) -> Dict[str, object]:
    raise NotImplementedError("Deprecated.")
    repo_dir = Path(repo_dir)
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise ValueError(f"repo_dir must be an existing directory, got: {repo_dir}")

    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", ".venv", "venv", "env", "build", "dist"}

    total_files = 0
    transformed_count = 0
    skipped_count = 0
    error_count = 0
    no_change_count = 0
    modified_files: List[str] = []
    error_map: Dict[str, str] = {}

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for fname in files:
            if not fname.endswith(".py"):
                continue

            file_path = Path(root) / fname
            if any(part in exclude_dirs for part in file_path.parts):
                skipped_count += 1
                continue

            total_files += 1
            try:
                src = file_path.read_text(encoding="utf-8")
            except Exception as e:
                error_count += 1
                error_map[str(file_path)] = f"read_error: {e!r}"
                continue

            try:
                new_src = transform_func(src)
            except Exception as e:
                error_count += 1
                error_map[str(file_path)] = f"transform_exception: {e!r}"
                continue

            if isinstance(new_src, str) and new_src != src:
                try:
                    st = file_path.stat()
                    file_path.write_text(new_src, encoding="utf-8")
                    os.chmod(file_path, st.st_mode)
                    transformed_count += 1
                    modified_files.append(str(file_path))
                except Exception as e:
                    error_count += 1
                    error_map[str(file_path)] = f"write_error: {e!r}"
                    continue
            else:
                no_change_count += 1

    return {
        "total_files": total_files,
        "transformed": transformed_count,
        "No change": no_change_count,
        "skipped": skipped_count,
        "errors": error_count,
        "modified_files": modified_files,
        "error_map": error_map,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-url", required=True, help="Git repo url to clone")
    p.add_argument(
        "--work-dir",
        required=True,
        help="Directory where repo will be cloned and test run",
    )
    p.add_argument(
        "--transformation",
        default=TransformationName.If_ELSE_SWAP.value,
        help="Transformations to apply",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="whether to apply Transformation in place",
    )
    args = p.parse_args()

    work_dir = Path(args.work_dir).absolute()
    project_root = Path(__file__).parent.absolute()
    report_dir = project_root / "test_reports"
    if not report_dir.exists():
        report_dir.mkdir(exist_ok=True)

    if work_dir.exists():
        print(f"Cleaning work dir {work_dir}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    repo_name = Path(args.repo_url).stem
    repo_dir = work_dir / repo_name
    print("Cloning repository...")
    git_clone(args.repo_url, repo_dir)

    print("Creating venv for original repo...")
    venv_dir = repo_dir / ".venv"
    make_venv(venv_dir)
    venv_py = venv_python_bin(venv_dir)

    print("Installing dependencies for original repo...")
    install_deps(repo_dir, venv_py)

    print("Running tests on original repo...")
    run_tests_with_coverage(
        repo_dir, report_dir, log_prefix=f"original_{repo_name}_", venv_python=venv_py
    )

    if args.transformation:
        transformation_name = args.transformation
        transformation_names = list(key.value for key in TransformationName)
        if transformation_name not in transformation_names:
            raise ValueError("Unknown transformation.")

        if not args.inplace:
            transformed_dir = work_dir / f"{repo_name}_transformed"
            print(f"Copying repository to {transformed_dir} ...")
            shutil.copytree(
                repo_dir,
                transformed_dir,
                ignore=shutil.ignore_patterns(
                    ".venv",
                    "*.pyc",
                    "__pycache__",
                    "pre_*",
                    "*_pytest.log",
                    "*_junit.xml",
                    "*_coverage.xml",
                    "*_htmlcov",
                ),
            )
            print("Copy complete.")

            print("Creating venv for transformed repo...")
            venv_dir = transformed_dir / ".venv"
            make_venv(venv_dir)
            venv_py = venv_python_bin(venv_dir)

            print("Installing dependencies for transformed repo...")
            install_deps(transformed_dir, venv_py)
        else:
            transformed_dir = repo_dir

        print("Applying transformation to repository files...")
        transform_directory(
            src_path=transformed_dir,
            transformation_name=transformation_name,
        )

        print("Running tests on transformed repo...")
        run_tests_with_coverage(
            transformed_dir,
            report_dir,
            log_prefix=f"transformed_{repo_name}_",
            venv_python=venv_py,
        )

    if args.inplace:
        delete_temp_files(work_dir)


if __name__ == "__main__":
    main()
