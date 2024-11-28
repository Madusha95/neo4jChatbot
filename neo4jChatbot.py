import os
import uuid
import logging
from datetime import datetime
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import Neo4jChatMessageHistory

# Set up logging (optional)
logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR)

# Utility function to generate unique session IDs
def get_session_id():
    """
    Generate a unique session ID for tracking chat sessions.
    This can be used to uniquely identify each session.
    """
    return str(uuid.uuid4())  # Generate a unique session ID

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Function to log errors
def log_error(error_message):
    logging.error(error_message)

# Fetch API credentials
openai_api_key = os.getenv("open_api_key")
openai_api_version = os.getenv("openai_api_version")

# Initialize the Neo4j graph connection
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="########",  # Ensure this is correct for your setup
    database="neo4j",
)

# Initialize the AzureChatOpenAI model
llm = AzureChatOpenAI(
    temperature=0.01, 
    streaming=True, 
    deployment_name="gpt-4o", 
    openai_api_version=openai_api_version, 
    openai_api_key=openai_api_key, 
    openai_api_base=openai_api_base
)

# Prompt template for Cypher query generation
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about products and provide recommendations.
Convert the user's question based on the schema provided.

Instructions:
- Use only the provided Nodes and Relationships in the schema.
- Use only the provided relationship types and properties in the schema.
- Cypher Query should start with "USE neo4j" every time.
- toLower function should be used for all attributes in the schema.
- and convert every attribute to string before using it in the query.
- always use `WHERE` clauses with `CONTAINS` for partial string matching (Don't use attributes names after `CONTAINS`)
- check the user input has a word in the schema such as Nodes,Relationships and Properties.

Schema:
Nodes:
- CompanyNode Labels and Properties:
  ['Customer']: ['CustomerCode', 'CustomerName']
  ['Supplier']: ['BuyerCode', 'Supplier','SupplierName']
  ['Product']: ['ItemCost', 'ItemType', 'PartDescription', 'PartNo', 'PriceWithoutTax']
 
Relationship Types and Properties:
- (Customer)-[:PLACES]->(SalesOrder)
- (SalesOrder)-[:INCLUDES_PRODUCT{{ OrderQty, ShippedQty, BOQty}}]->(Product)
- (SalesOrder)-[:DISPATCHED_BY]->(Warehouse)
- (Warehouse)-[:DELIVERS_TO]->(Customer)
- (Company)-[:OWNS]->(Warehouse)
- (Company)-[:ISSUES]->(PurchaseOrder)

Question: {question}

"""

# Create tools for the agent with error handling
def cypher_qa(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "There was an issue processing your request. Please try again or contact support."

def order_tracking(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "Unable to track the order at the moment. Please try again later or contact support."

def supplier_info(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "Unable to fetch supplier information. Please try again later or contact support."

def warehouse_inventory(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "Unable to fetch warehouse inventory at the moment. Please try again later or contact support."

def purchase_order_info(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "Unable to retrieve purchase order information. Please try again later or contact support."

def customer_info(query):
    try:
        result = cypher_chain.invoke({"query": query})
        return result
    except Exception as e:
        log_error(f"Cypher error occurred: {str(e)}")
        print(e)
        return "Unable to retrieve customer information. Please try again later or contact support."

tools = [
    Tool.from_function(
        name="Product Information",
        description="Answer product-related and warehouse-related questions using Cypher in the Neo4j database.",
        func=cypher_qa,
    ),
    Tool.from_function(
        name="Order Tracking",
        description="Help track orders based on customer input using Cypher queries in the Neo4j database.",
        func=order_tracking,
    ),
    Tool.from_function(
        name="Supplier Information",
        description="Fetch supplier-related information using Cypher queries in the Neo4j database.",
        func=supplier_info,
    ),
    Tool.from_function(
        name="Warehouse Inventory",
        description="Fetch the status of warehouse inventory using Cypher queries in the Neo4j database.",
        func=warehouse_inventory,
    ),
    Tool.from_function(
        name="Purchase Order Information",
        description="Retrieve information related to purchase orders using Cypher queries in the Neo4j database.",
        func=purchase_order_info,
    ),
    Tool.from_function(
        name="Customer Information",
        description="Retrieve customer-related information using Cypher queries in the Neo4j database.",
        func=customer_info,
    ),
    Tool.from_function(
        name="General Chat",
        description="For general questions or unstructured queries.",
        func=llm.invoke,
    )
]

# Get current date components
current_year = datetime.now().year
current_month_name = datetime.now().strftime("%B")

# Create agent prompt
agent_prompt = PromptTemplate.from_template(f"""
You are an expert in answering product-related questions by accessing a Neo4j database and generating Cypher queries.
When the user asks about products, sales, orders, warehouses, suppliers, or customers, use the corresponding tool.
If the question is general or does not require database access, use 'General Chat'.
limit output to maximum 5 responses.
Limit responses to the **final answer only**. Do not include additional action metadata in the response.                                            

Date Interpretation Guidelines:
1. When the input contains "this month," interpret it as "{current_month_name} {current_year}".
2. For "this week" or "today," use the specific date range based on the current date.
3. If a specific month is mentioned without a year, interpret it as "{current_year}" unless the year is specified otherwise.
4. If the input is unclear about the date, ask the user to clarify, specifying both month and year.

TOOLS:
------
You have access to the following tools:
{{tools}}

To use a tool, follow this format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, respond in this format:
Thought: Do I need to use a tool? No

Logic for Final Answer:
- If the user input matches with specific schema terms, identify relevant nodes, properties, and relationships and construct the Cypher query accordingly.
- If input is a simple greeting like "Hello" or "Hi", respond with the generated Final Answer: your response here.
- For simple numbers, prompt for specifics related to products, sales, warehouses, orders, or suppliers.
- For sensitive inquiries, respond with:
  Final Answer: Please contact ABC Customer Service for assistance with products, sales, warehouses, orders, or suppliers.
- For date-related questions with terms like "this month" or "{current_month_name}", automatically substitute with the appropriate month and year.
- If none of the above conditions are met, respond with:
  Final Answer: Could you please provide more details related to products, sales, warehouses, orders, or suppliers at ABC?

Previous conversation history:
{{chat_history}}

New input: {{input}}
{{agent_scratchpad}}
""")

# Initialize the Graph Cypher QA Chain
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=PromptTemplate(template=CYPHER_GENERATION_TEMPLATE, input_variables=["question"]),
    verbose=True,
    allow_dangerous_requests=True, # Allow dangerous requests
    temperature=0.01, # Set the temperature
)

# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, allow_dangerous_requests=True)

# Create RunnableWithMessageHistory for message history tracking
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Function to generate response based on input
def generate_response(user_input):
    """
    Create a handler that calls the conversational agent
    and returns a response
    """
    try:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}}
        )
        print(response)
        
        # Extract the Final Answer from the chain execution
        if response['output']:
            final_answer = response['output']
        else:
            final_answer = "I couldn't find the exact information in the database. Please try again or contact support."
        
        return final_answer
    except Exception as e:
        # Log the error (optional)
        log_error(f"Error generating response: {str(e)}")
        # Return a generic error message
        return "There was an issue processing your request. Please try again later or contact support."

# Console-based chat loop
def chat_loop():
    print("Hello! I am your product chatbot. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Start the chat loop
if __name__ == "__main__":
    chat_loop()
