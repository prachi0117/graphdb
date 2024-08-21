import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://2f39d47e.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "_YGbxuqFNJevkNthWRnhpGCxsSfwgTy_k65IxKmNZu0")

# Set up Groq connection
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Neo4j Graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# Initialize the GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# Streamlit UI
st.title("Neo4j Movie Database Query App")

# Session state for query history
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Input query from user
query = st.text_input("Enter your query", "Who was the director of the movie Casino?")

# Display schema

# Process the query and display the response
if st.button("Run Query"):
    with st.spinner("Running query..."):
        try:
            response = chain.invoke({"query": query})
            # Extract the result from the response
            result = response.get('result', 'No result returned.')
            if result.strip().lower() == "i don't know the answer.":
                st.warning("The model couldn't find an answer to your query.")
            else:
                st.success("Query executed successfully!")
                st.write("Response:")
                st.write(result)
                
                # Save the query and result to history
                st.session_state.query_history.append({"query": query, "result": result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Show query history
if st.button("Show Query History"):
    if st.session_state.query_history:
        st.write("Query History:")
        for idx, entry in enumerate(st.session_state.query_history, 1):
            st.write(f"{idx}. **Query:** {entry['query']}")
            st.write(f"   **Result:** {entry['result']}")
    else:
        st.write("No queries run yet.")

# Optionally, add a file uploader for more advanced features
# uploaded_file = st.file_uploader("Upload a CSV file to load data into Neo4j", type="csv")
# if uploaded_file is not None:
#     # Process file here (not implemented in this example)
#     st.write("File uploaded successfully!")
