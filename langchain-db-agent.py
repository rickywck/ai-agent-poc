
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

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
#from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain.agents.agent_types import AgentType

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

from langchain_community.llms import OpenAI

# Initialize the LLM
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
#llm = ChatOllama(model="deepseek-r1:8b")
llm = ChatOllama(model="tulu3:8b")
#llm = ChatOllama(model="qwen2.5:7b")
#llm = ChatOllama(model="llama3.1:8b")

#llm = ChatMistralAI(
#    model="mistral-small-latest",
#    temperature=0,
#    max_retries=2,
#    # other params...
#)

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit

# Create a toolkit for the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=20
)

#query = "What is the price of 'ABC' stock on January 1, 2023?"
#response = agent_executor.invoke(query)
#print(response)

#query = "What are the stock prices for 'ABC' and 'XYZ' on January 3rd and January 4th?"
#response = agent_executor.invoke(query)
#print(response)


import sys

print("Enter the question:")
query = sys.stdin.readline().strip()
while True:
    response = agent_executor.invoke(query)
    print(response)
    print("Next question:")
    query = sys.stdin.readline().strip()
    if query == "exit":
        break

