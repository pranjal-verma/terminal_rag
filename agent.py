from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})

def node_a(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

def node_b(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph.add_node(node_a)
graph.add_node(node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)