"""Credential-safe access check. Does not log in or persist tokens.

--token-prompt reads from a hidden terminal prompt; --download is a separate
opt-in, pinned download of only the model files needed by the feature worker.
"""
import argparse
import getpass
import json
from pathlib import Path
import urllib.error
import urllib.request

from .io import atomic_json

REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


def check(token=None):
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/config.json"
    request = urllib.request.Request(url, method="HEAD", headers={"Authorization": "Bearer " + token} if token else {})
    result = dict(repo=REPO, revision=REVISION, credential_supplied=bool(token))
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result.update(status=response.status, granted=response.status == 200)
    except urllib.error.HTTPError as exc:
        result.update(status=exc.code, granted=False, error_code=exc.headers.get("X-Error-Code"))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token-prompt", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--cache-dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    token = getpass.getpass("Hugging Face token (not saved): ") if args.token_prompt else None
    result = check(token)
    if args.download and result["granted"]:
        if not args.cache_dir:
            raise ValueError("provide a separate encoder cache directory")
        from huggingface_hub import snapshot_download
        result["snapshot"] = snapshot_download(REPO, revision=REVISION, cache_dir=args.cache_dir, token=token,
                                                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    atomic_json(args.out, result)
    print(json.dumps(result))
    if not result["granted"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
