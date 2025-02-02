
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# SQLAlchemy setup
from sqlalchemy import MetaData, Column, Integer, String, Table, Date, Float, create_engine
from sqlalchemy import insert
from datetime import datetime

metadata_obj = MetaData()

# Define the database table
stocks = Table(
    "stocks",
    metadata_obj,
    Column("obs_id", Integer, primary_key=True),
    Column("stock_name", String(4), nullable=False),
    Column("price", Float, nullable=False),
    Column("date", Date, nullable=False),
)

# Create in-memory SQLite database
engine = create_engine("sqlite:///:memory:")
metadata_obj.create_all(engine)

# Insert data into the database
observations = [
    [1, 'ABC', 200, datetime(2023, 1, 1)],
    [2, 'ABC', 208, datetime(2023, 1, 2)],
    [3, 'ABC', 232, datetime(2023, 1, 3)],
    [4, 'ABC', 225, datetime(2023, 1, 4)],
    [5, 'ABC', 226, datetime(2023, 1, 5)],
    [6, 'XYZ', 810, datetime(2023, 1, 1)],
    [7, 'XYZ', 803, datetime(2023, 1, 2)],
    [8, 'XYZ', 798, datetime(2023, 1, 3)],
    [9, 'XYZ', 795, datetime(2023, 1, 4)],
    [10, 'XYZ', 791, datetime(2023, 1, 5)],
]

def insert_obs(obs):
    stmt = insert(stocks).values(
        obs_id=obs[0],
        stock_name=obs[1],
        price=obs[2],
        date=obs[3]
    )
    with engine.begin() as conn:
        conn.execute(stmt)

for obs in observations:
    insert_obs(obs)

# print database created 
print("Database created successfully ...")

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, Graph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from operator import itemgetter
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

# Initialize the LLM
llm = ChatOllama(model="tulu3:8b")

# Create SQL database tool
sql_tool = QuerySQLDataBaseTool(db=db)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next: str

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful SQL assistant that helps users query stock price information.
    Use the provided SQL tool to answer questions about stock prices.
    Always explain your reasoning before executing queries."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the nodes for our graph
def agent(state: AgentState):
    messages = state["messages"]
    
    # Generate response using the LLM
    response = llm.invoke(prompt.invoke({"messages": messages}))
    
    # If we need to query the database
    if "SELECT" in response.content.upper():
        # Extract the SQL query
        query = response.content[response.content.find("SELECT"):].split(";")[0]
        try:
            result = sql_tool.invoke(query)
            return {"messages": messages + [response] + [("system", f"Query result: {result}")], "next": END}
        except Exception as e:
            return {"messages": messages + [response] + [("system", f"Error: {str(e)}")], "next": END}
    
    return {"messages": messages + [response], "next": END}

# Create the graph
workflow = Graph()
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")

# Compile the graph
chain = workflow.compile()

# Example query
query = "What are the stock prices for 'ABC' and 'XYZ' on January 3rd and January 4th?"
response = chain.invoke({
    "messages": [("human", query)]
})
print("\nFinal Response:")
for message in response["messages"]:
    print(f"{message.type}: {message.content}")



