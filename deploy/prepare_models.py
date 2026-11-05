"""Explicit model preparation into the Compose cache volume; never run during build."""
import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Also download the visual embedding (large)")
    args = parser.parse_args()
    models = [(os.getenv("EMBEDDING_LOCAL_MODEL", "intfloat/multilingual-e5-small"), None),
              (os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"), None)]
    if args.visual:
        models.append((os.getenv("VISUAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
                       os.getenv("VISUAL_EMBEDDING_REVISION", "main")))
    for model, revision in models:
        if os.path.isdir(model):
            print("Using mounted model directory:", model, flush=True)
        else:
            print("Downloading model:", model, flush=True)
            snapshot_download(model, revision=revision)


if __name__ == "__main__":
    main()
