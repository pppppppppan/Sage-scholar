import logging
import os
from openai import OpenAI, APIConnectionError, APIStatusError
from dotenv import load_dotenv
from .local_embedding import LocalEmbedding, DEFAULT_MODEL

logger = logging.getLogger(__name__)

load_dotenv()


class Embedding:
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
        provider: str = None,
    ):
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER", "api")).lower()
        self.local = None
        if self.provider == "local":
            self.model = model or os.getenv("EMBEDDING_LOCAL_MODEL", DEFAULT_MODEL)
            self.local = LocalEmbedding(self.model)
            self.client = None
            return
        if self.provider != "api":
            raise ValueError("EMBEDDING_PROVIDER 只支持 local 或 api")
        self.model = model or os.getenv("EMBEDDING_MODEL_ID")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("GLM_API_KEY")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        self.timeout = timeout or int(os.getenv("EMBEDDING_TIMEOUT", 60))

        if not all([self.model, self.api_key, self.base_url]):
            raise ValueError(
                "model_id, api_key and base_url have to be predefined or given."
            )

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    @property
    def identity(self):
        if self.local is not None:
            return self.local.identity
        return {"provider": "api", "model": self.model, "base_url": self.base_url}

    MAX_BATCH_SIZE = 64

    def invoke(self, input_text: list[str] | str, *, task: str | None = None) -> tuple[list, int]:
        is_single = isinstance(input_text, str)
        if is_single:
            input_text = [input_text]

        task = task or ("query" if is_single else "passage")
        if task not in {"query", "passage"}:
            raise ValueError("embedding task must be query or passage")
        if self.local is not None:
            vectors = self.local.invoke(input_text, task=task)
            # This counter tracks remote API usage; local inference consumes no API tokens.
            return (vectors[0] if is_single else vectors, 0)

        logger.info("调用 %s 向量模型, %d 条输入", self.model, len(input_text))

        all_embeddings = []
        total_tokens = 0
        for i in range(0, len(input_text), self.MAX_BATCH_SIZE):
            batch = input_text[i : i + self.MAX_BATCH_SIZE]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
            except APIStatusError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error", body)
                code = str(error.get("code", "")) if isinstance(error, dict) else ""
                logger.warning("Embedding request failed: model=%s status=%s code=%s",
                               self.model, exc.status_code, code)
                if code == "1211" or code == "model_not_found":
                    detail = f"向量化失败：服务端不接受模型 {self.model}。请核对 EMBEDDING_MODEL_ID、EMBEDDING_BASE_URL，以及该 Key 是否可使用这个向量模型。"
                elif exc.status_code in {401, 403}:
                    detail = "向量化失败：API Key 无效或没有访问权限，请检查 EMBEDDING_API_KEY（未配置时使用 GLM_API_KEY）。"
                elif exc.status_code == 429:
                    detail = "向量化失败：服务限流或额度不足，请检查向量服务账户后重试。"
                else:
                    detail = f"向量化失败：向量服务返回 HTTP {exc.status_code}，请检查模型配置和输入限制。"
                raise RuntimeError(detail) from exc
            except APIConnectionError as exc:
                raise RuntimeError("向量化失败：无法连接向量服务或请求超时，请检查网络及 EMBEDDING_BASE_URL 后重试。") from exc
            total_tokens += response.usage.total_tokens
            all_embeddings.extend(item.embedding for item in response.data)

        return (all_embeddings[0] if is_single else all_embeddings, total_tokens)
