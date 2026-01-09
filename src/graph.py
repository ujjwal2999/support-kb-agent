"""
LangGraph RAG workflow

Simple two-node graph: retrieve -> generate
Thought about adding a reranking step but keeping it simple for now
"""
from typing import TypedDict, List
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from src.config import GEMINI_API_KEY, LLM_MODEL
from src.store import store

# new google-genai client setup
client = genai.Client(api_key=GEMINI_API_KEY)


class AgentState(TypedDict):
    query: str
    retrieved_docs: List[dict]
    answer: str


def retrieve_node(state: AgentState) -> AgentState:
    """get relevant docs from vector store"""
    query = state["query"]
    docs = store.search(query, k=3)
    return {**state, "retrieved_docs": docs}


def generate_node(state: AgentState) -> AgentState:
    """generate answer using retrieved context"""
    docs = state["retrieved_docs"]
    
    if not docs:
        return {**state, "answer": "Sorry, I couldn't find relevant info in the knowledge base."}
    
    # build context string
    context_parts = []
    for doc in docs:
        context_parts.append(f"[Source: {doc['source']}]\n{doc['content']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are a helpful support agent. Answer the user's question based ONLY on the 
following documents from the knowledge base. If the documents don't contain the answer, 
say so - don't make things up.

DOCUMENTS:
{context}

USER QUESTION: {state['query']}

Provide a helpful answer and cite which document(s) you used."""
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        answer_text = response.text
    except Exception as e:
        answer_text = f"Error generating response: {str(e)}"
    
    sources = list(set(doc["source"] for doc in docs))
    final_answer = f"{answer_text}\n\n---\nSources: {', '.join(sources)}"
    
    return {**state, "answer": final_answer}


def build_rag_graph():
    """build and compile the langgraph workflow"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def query(question: str) -> str:
    """run a query through the rag pipeline"""
    graph = build_rag_graph()
    
    initial_state: AgentState = {
        "query": question,
        "retrieved_docs": [],
        "answer": ""
    }
    
    result = graph.invoke(initial_state)
    return result["answer"]
