"""Upload the trained model to the Hugging Face Hub.

Requires HF_TOKEN to be set in the environment (a token with write access).
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Ambar10/brain-tumor-efficientnetb0"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(ARTIFACTS_DIR / "efficientnetb0_model.keras"),
        path_in_repo="efficientnetb0_model.keras",
        repo_id=REPO_ID,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(ARTIFACTS_DIR / "class_names.json"),
        path_in_repo="class_names.json",
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded model + class names to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
