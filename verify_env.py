"""End-to-end Windows GPU smoke test for the rag-agent-gpu environment.

Run:  <env>\\python.exe verify_env.py
"""
import os
import sys
import time

# Must be set before torch / transformers import.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass
os.environ.setdefault("HF_HOME", r"C:\hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console
from rich.table import Table

console = Console()
results = []  # (check, status, detail)


def record(name, ok, detail):
    results.append((name, "PASS" if ok else "FAIL", str(detail)))


def run(name):
    def deco(fn):
        try:
            record(name, True, fn())
        except Exception as exc:  # noqa: BLE001
            record(name, False, f"{type(exc).__name__}: {exc}")
        return fn

    return deco


SAMPLE_DOCS = [
    "Retrieval-augmented generation grounds LLM answers in external documents.",
    "FAISS performs fast similarity search over dense vectors.",
    "LangGraph models agent workflows as stateful graphs.",
    "The RTX 3090 has 24 GB of VRAM, enough for 30B-class quantized models.",
    "BM25 is a sparse lexical ranking function used in hybrid search.",
]
STATE = {}


@run("1. OS & environment")
def _():
    assert sys.platform == "win32", f"platform is {sys.platform}"
    assert sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]}"
    in_env = "rag-agent-gpu" in sys.executable
    hf = os.environ.get("HF_HOME")
    try:
        import winreg

        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        longp = winreg.QueryValueEx(k, "LongPathsEnabled")[0]
    except Exception:
        longp = "?"
    return (
        f"win32 | py {sys.version.split()[0]} | env={'rag-agent-gpu' if in_env else 'NOT rag-agent-gpu'} "
        f"| HF_HOME={hf} | LongPaths={longp}"
    )


@run("2. CUDA & PyTorch (FP16 matmul)")
def _():
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    props = torch.cuda.get_device_properties(0)
    a = torch.randn(2048, 2048, device="cuda:0", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda:0", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    assert torch.isfinite(c).all().item(), "non-finite matmul result"
    return (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) | {props.name} | "
        f"{props.total_memory / 1024**3:.1f} GB | matmul {ms:.1f} ms"
    )


@run("3. GPU embeddings (bge-small-en-v1.5)")
def _():
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda:0")
    model.encode(SAMPLE_DOCS, normalize_embeddings=True)  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    emb = model.encode(SAMPLE_DOCS, normalize_embeddings=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    STATE["emb"] = emb
    STATE["model"] = model
    mem = torch.cuda.memory_allocated() / 1024**2
    return f"device={model.device} | shape={emb.shape} | {ms:.1f} ms | VRAM alloc {mem:.0f} MB"


@run("4. Vector search (FAISS + Chroma)")
def _():
    import faiss
    import numpy as np

    emb = np.asarray(STATE["emb"], dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    q = STATE["model"].encode(["How are agents orchestrated?"], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), 2)
    top = SAMPLE_DOCS[ids[0][0]]
    assert "LangGraph" in top, f"unexpected top hit: {top}"

    import chromadb

    client = chromadb.EphemeralClient()
    col = client.create_collection("smoke")
    col.add(ids=[str(i) for i in range(len(SAMPLE_DOCS))], documents=SAMPLE_DOCS, embeddings=emb.tolist())
    res = col.query(query_embeddings=q.tolist(), n_results=1)
    return f"faiss top1='{top[:34]}...' ({scores[0][0]:.2f}) | chroma top1 id={res['ids'][0][0]}"


@run("5. LangGraph StateGraph + mock tool")
def _():
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from pydantic import BaseModel
    import pydantic

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    class State(BaseModel):
        a: int = 0
        b: int = 0
        result: int | None = None

    def call_tool(s: State):
        return {"result": add.invoke({"a": s.a, "b": s.b})}

    g = StateGraph(State)
    g.add_node("tool", call_tool)
    g.add_edge(START, "tool")
    g.add_edge("tool", END)
    out = g.compile().invoke({"a": 2, "b": 3})
    assert out["result"] == 5, out
    return f"graph result={out['result']} | pydantic {pydantic.VERSION}"


table = Table(title="rag-agent-gpu smoke test", show_lines=True)
table.add_column("Check", style="bold")
table.add_column("Status")
table.add_column("Detail", overflow="fold")
for name, status, detail in results:
    table.add_row(name, f"[green]{status}[/]" if status == "PASS" else f"[red]{status}[/]", detail)
console.print(table)
sys.exit(0 if all(s == "PASS" for _, s, _ in results) else 1)
