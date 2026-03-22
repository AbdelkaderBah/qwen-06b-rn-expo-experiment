import httpx
import subprocess
from pathlib import Path

SOURCES = {
    "react-native": {
        "llms": "https://reactnative.dev/llms-full.txt",
        "git": ("https://github.com/facebook/react-native-website", "docs"),
    },
    "expo": {
        "llms": "https://expo.dev/llms-full.txt",
        "git": ("https://github.com/expo/expo", "docs/pages"),
    },
    "react-navigation": {
        "llms": "https://reactnavigation.org/llms-full.txt",
        "git": ("https://github.com/react-navigation/react-navigation.github.io", "versioned_docs/version-7.x"),
    },
    "reanimated": {
        "llms": "https://docs.swmansion.com/react-native-reanimated/llms-full.txt",
        "git": ("https://github.com/software-mansion/react-native-reanimated", "docs/docs-reanimated"),
    },
}

RAW = Path("data/raw")


def fetch_llms_txt(name: str, url: str) -> bool:
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        if r.status_code == 200:
            (RAW / f"{name}.txt").write_text(r.text)
            print(f"✅ {name} — llms.txt ({len(r.text) // 1000}kb)")
            return True
        print(f"⚠️  {name} — llms.txt returned HTTP {r.status_code}")
    except Exception as e:
        print(f"⚠️  {name} — llms.txt failed: {e}")
    return False


def fetch_git_sparse(name: str, repo: str, folder: str):
    dest = RAW / f"{name}_git"
    if dest.exists():
        print(f"⏭️  {name} — git folder already exists, skipping")
        return
    dest.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", repo, str(dest)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", folder], cwd=dest, check=True)
    print(f"✅ {name} — git sparse-checkout done → {dest}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, sources in SOURCES.items():
        if not fetch_llms_txt(name, sources["llms"]):
            fetch_git_sparse(name, *sources["git"])
