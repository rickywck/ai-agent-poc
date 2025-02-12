import getpass
import os


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("ANTHROPIC_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Ask for LLM mode
llm_mode = input("Select LLM mode (OpenAI/Ollama/Gemini): ").strip().lower() or "openai"
if llm_mode == "openai":
    model = input("Enter OpenAI model (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model, temperature=0, verbose=True)
elif llm_mode == "ollama":
    model = input("Enter Ollama model (default: qwen2.5:7b): ").strip() or "qwen2.5:7b"
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model)
    print(f"Using Ollama model: {model}")
elif llm_mode == "gemini":
    model = input("Enter Gemini model (default: gemini-2.0-flash): ").strip() or "gemini-2.0-flash"
    from langchain_google_vertexai import ChatVertexAI
    llm = ChatVertexAI(model=model, temperature=0, max_tokens=None, max_retries=6, stop=None)
    print(f"Using Gemini model: {model}")
else:
    raise ValueError("Invalid LLM mode selected. Please choose either 'OpenAI', 'Ollama', or 'Gemini'.")

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_vertexai import ChatVertexAI
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, END
from langgraph.types import Command


members = ["researcher", "coder"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["researcher", "coder", "FINISH"]


#llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
#llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
#llm = ChatOllama(model="qwen2.5:7b")
#llm = ChatVertexAI(model="gemini-2.0-flash",temperature=0,max_tokens=None,max_retries=6,stop=None)

class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal["researcher", "coder", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
    except TypeError as e:
        print(f"Error: {e}")
        print("LLM response:", response)
        exit(1)
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


research_agent = create_react_agent(
    llm, tools=[tavily_tool], prompt="You are a researcher. DO NOT do any math."
)


def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_react_agent(llm, tools=[python_repl_tool])


def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
graph = builder.compile()

# no need to show the graph when the code is running in command line
#from IPython.display import display, Image
#display(Image(graph.get_graph().draw_mermaid_png()))

for s in graph.stream(
    {"messages": [("user", "What's the square root of 42?")]}, subgraphs=True
):
    print(s)
    print("----")


for s in graph.stream(
    {
        "messages": [
            (
                "user",
                "Find the latest GDP of New York and California, then calculate the average",
            )
        ]
    },
    subgraphs=True,
):
    print(s)
    print("----")