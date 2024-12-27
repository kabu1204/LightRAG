import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding, ollama_model_complete, zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from .code import chunking_code_repo
from FlagEmbedding import BGEM3FlagModel
embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, cache_dir="./models") # Setting use_fp16 to True speeds up computation with a slight performance degradation

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

DEEPSEEK_API_KEY="sk-9a1e1c0614c1449a9c9187be78f396b6"
ZHIPUAI_API_KEY="d3ccd56132e7860342f58ef021bb49d7.Anr4vxEaSSw79NOV"

async def deepseek_llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        **kwargs,
    )

async def zhipu_llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flashx",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=ZHIPUAI_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        **kwargs,
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return embedding_model.encode(texts)['dense_vecs']

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print("embedding dim:", embedding_dim)
    return embedding_dim

# function test
async def test_funcs():
    result = await deepseek_llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)

CODE_ENTITY_TYPES = ["class", "function", "variable", "argument", "file"]

REPO_DESC = '''
{
    "name": "lightrag",
    "src": ["lightrag", "examples"],
    "doc": ["README.md"]
}
'''

async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=deepseek_llm_model_func,
            # llm_model_func=ollama_model_complete,
            # llm_model_name="qwen2.5:1.5b",
            # llm_model_max_async=4,
            # llm_model_max_token_size=10240,
            # llm_model_kwargs={"host": "http://137.189.89.85:11434", "options": {"num_ctx": 10240}},
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
            chunking_func=chunking_code_repo,
            addon_params={
                "entity_types": CODE_ENTITY_TYPES
            }
        )

        await rag.ainsert(REPO_DESC)

        # Perform naive search
        # print(
        #     await rag.aquery(
        #         "Introduce the repo", param=QueryParam(mode="naive")
        #     )
        # )

        # Perform local search
        print(
            await rag.aquery(
                "Introduce the repo", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        # print(
        #     await rag.aquery(
        #         "Introduce the repo",
        #         param=QueryParam(mode="global"),
        #     )
        # )

        # Perform hybrid search
        # print(
        #     await rag.aquery(
        #         "Introduce the repo",
        #         param=QueryParam(mode="hybrid"),
        #     )
        # )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())

    # asyncio.run(get_embedding_dim())