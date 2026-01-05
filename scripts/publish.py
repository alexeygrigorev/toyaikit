#!/usr/bin/env python3
"""
Release script for toyaikit.

Runs tests, bumps version, and publishes to PyPI (dev and/or prod).

Usage:
    python scripts/publish.py               # Bump patch version, publish to dev and prod
    python scripts/publish.py --minor       # Bump minor version
    python scripts/publish.py --major       # Bump major version
    python scripts/publish.py --version 0.1.5  # Set specific version
    python scripts/publish.py --dev-only    # Only publish to testpypi
    python scripts/publish.py --prod-only   # Only publish to prod pypi
    python scripts/publish.py --skip-tests  # Skip running tests
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd, check=True, shell=False, cwd=None):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        check=check,
        shell=shell,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.stdout:
        print(result.stdout)
    return result


def check_git_clean():
    """Check if there are no uncommitted changes in toyaikit directory."""
    project_root = Path(__file__).parent.parent
    result = run(
        ["git", "status", "--porcelain", "toyaikit/"],
        cwd=project_root,
        check=False,
    )
    if result.stdout.strip():
        print("ERROR: There are uncommitted changes in toyaikit/:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("\nPlease commit or stash your changes before publishing.", file=sys.stderr)
        sys.exit(1)


def check_tag_not_exists(new_version):
    """Check if the tag already exists."""
    project_root = Path(__file__).parent.parent
    result = run(
        ["git", "tag", "-l", f"{new_version}"],
        cwd=project_root,
        check=False,
    )
    if result.stdout.strip():
        print(f"ERROR: Tag {new_version} already exists.", file=sys.stderr)
        print("\nTo re-release this version, first delete the existing tag:", file=sys.stderr)
        print(f"  git tag -d {new_version}", file=sys.stderr)
        sys.exit(1)


def get_current_version():
    """Get the current version from __version__.py."""
    version_file = Path(__file__).parent.parent / "toyaikit" / "__version__.py"
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find version in {version_file}")
    return match.group(1)


def bump_version(current, bump_type=None, new_version=None):
    """Bump version according to type or set to specific version."""
    if new_version:
        return new_version

    parts = current.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current}")

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    return f"{major}.{minor}.{patch}"


def set_version(new_version):
    """Write the new version to __version__.py."""
    version_file = Path(__file__).parent.parent / "toyaikit" / "__version__.py"
    content = version_file.read_text()
    new_content = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content,
    )
    version_file.write_text(new_content)
    print(f"Updated version to {new_version}")


def run_tests():
    """Run the test suite."""
    print("\n=== Running tests ===")
    project_root = Path(__file__).parent.parent
    result = run(
        ["make", "test"],
        cwd=project_root,
        check=False,
    )
    if result.returncode != 0:
        print("Tests failed!", file=sys.stderr)
        sys.exit(1)
    print("Tests passed!")


def build_package():
    """Build the package."""
    print("\n=== Building package ===")
    project_root = Path(__file__).parent.parent
    run(["make", "publish-build"], cwd=project_root)


def publish_to(repository_name):
    """Publish package to a specific repository (uses .pypirc config)."""
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"

    # Find the wheel and tar.gz files
    wheels = list(dist_dir.glob("*.whl"))
    tarballs = list(dist_dir.glob("*.tar.gz"))

    if not wheels or not tarballs:
        raise FileNotFoundError("No wheel or tar.gz files found in dist/")

    # Publish using repository name from .pypirc
    files_to_upload = [str(wheels[0]), str(tarballs[0])]
    run(
        ["uv", "run", "twine", "upload", "--repository", repository_name]
        + files_to_upload,
        cwd=project_root,
    )


def clean_dist():
    """Clean the dist directory."""
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        for file in dist_dir.glob("*"):
            file.unlink()


def main():
    parser = argparse.ArgumentParser(description="Release script for toyaikit")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--major", action="store_true", help="Bump major version"
    )
    group.add_argument(
        "--minor", action="store_true", help="Bump minor version"
    )
    group.add_argument(
        "--patch", action="store_true", help="Bump patch version (default)"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Set specific version (overrides --major/--minor/--patch)",
    )
    parser.add_argument(
        "--dev-only", action="store_true", help="Only publish to testpypi"
    )
    parser.add_argument(
        "--prod-only", action="store_true", help="Only publish to prod pypi"
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests"
    )

    args = parser.parse_args()

    # Check for uncommitted changes
    check_git_clean()

    # Determine bump type
    bump_type = "patch"
    if args.major:
        bump_type = "major"
    elif args.minor:
        bump_type = "minor"

    # Get and bump version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    new_version = bump_version(current_version, bump_type, args.version)
    print(f"New version: {new_version}")

    # Check if tag already exists
    check_tag_not_exists(new_version)

    # Run tests
    if not args.skip_tests:
        run_tests()

    # Set new version
    set_version(new_version)

    # Build
    clean_dist()
    build_package()

    # Publish
    if not args.prod_only:
        print("\n=== Publishing to TestPyPI ===")
        publish_to("testpypi")

    if not args.dev_only:
        print("\n=== Publishing to PyPI ===")
        publish_to("pypi")

    # Commit and tag (only after successful publish)
    print("\n=== Committing and tagging ===")
    project_root = Path(__file__).parent.parent
    run(["git", "add", "toyaikit/__version__.py"], cwd=project_root)
    run(["git", "commit", "-m", f"Bump version to {new_version}"], cwd=project_root)
    run(["git", "tag", f"{new_version}"], cwd=project_root)

    # Push tag
    print("\n=== Pushing to GitHub ===")
    run(["git", "push"], cwd=project_root)
    run(["git", "push", "--tags"], cwd=project_root)

    print(f"\n=== Released version {new_version} successfully! ===")


if __name__ == "__main__":
    main()
