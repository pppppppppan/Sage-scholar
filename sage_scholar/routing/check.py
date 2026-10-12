"""python -m sage_scholar.routing.check --query '不要调研，只解释 KL 散度'"""
import argparse
import json
from sage_scholar.config import Config
from .classifier import LocalClassifier, RouterUnavailable
from .router import QueryRouter


def main():
    parser = argparse.ArgumentParser(description="真实调用本机 Ollama Router，不调用云端 LLM 或 MCP")
    parser.add_argument("--query", default="不要调研，只解释一下 KL 散度")
    args = parser.parse_args()
    config = Config().router
    try:
        route = QueryRouter(LocalClassifier(config), confidence_threshold=config.confidence_threshold).route(args.query)
    except (RouterUnavailable, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"model": config.model, "route": route.model_dump(mode="json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
