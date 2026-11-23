"""LangChain smoke test for local Qwen served by LM Studio."""
import sys, time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",
                 model="qwen3-coder-30b-a3b-instruct", temperature=0, timeout=300)


def invoke():
    r = llm.invoke("Reply with exactly: pong")
    assert "pong" in r.content.lower(), r.content
    return repr(r.content)


def lcel_chain():
    chain = (ChatPromptTemplate.from_template("Give a one-sentence definition of {term}.")
             | llm | StrOutputParser())
    out = chain.invoke({"term": "vector database"})
    assert len(out) > 20
    return out


def streaming():
    chunks = [c.content for c in llm.stream("Count from 1 to 5.") if c.content]
    assert chunks
    return f"{len(chunks)} chunks: {''.join(chunks)!r}"


class City(BaseModel):
    name: str = Field(description="City name")
    country: str


def structured():
    r = llm.with_structured_output(City).invoke("Paris is the capital of France.")
    assert r.name.lower() == "paris", r
    return repr(r)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def tool_calling():
    msg = llm.bind_tools([multiply]).invoke("What is 6 times 7? Use the tool.")
    call = msg.tool_calls[0]
    assert call["name"] == "multiply"
    return f"{call['name']}({call['args']}) -> {multiply.invoke(call['args'])}"


def agent():
    from langchain.agents import create_agent
    a = create_agent(llm, tools=[multiply])
    out = a.invoke({"messages": [("user", "Use the tool to compute 12 * 11.")]})
    final = out["messages"][-1].content
    assert "132" in final, final
    return final


def run(name, fn):
    t = time.perf_counter()
    try:
        detail = fn()
        print(f"[PASS] {name} ({time.perf_counter()-t:.1f}s): {detail}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    tests = [("invoke", invoke), ("LCEL chain", lcel_chain), ("streaming", streaming),
             ("structured output", structured), ("tool calling", tool_calling), ("agent", agent)]
    sys.exit(0 if all([run(n, f) for n, f in tests]) else 1)
