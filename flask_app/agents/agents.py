from langchain.agents.middleware import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def agent_generation(prompt,model='gpt-5-nano', reasoning_effort='medium', tools:list = None, checkpointer=None, middleware=()):
    # Inicializa el LLM
    selector_model = ChatOpenAI(
        model=model,
        reasoning_effort=reasoning_effort)

    selector_agent = create_agent(
        model=selector_model,
        tools=tools,
        system_prompt=prompt,
        middleware=middleware,
        checkpointer=checkpointer
    )

    return selector_agent

def middleware_generation(model:str = "gpt-5-nano", reasoning_effort="low", max_tokens_before_summary:int =2000, messages_to_keep:int=2):
    summarization_prompt = (
    "You are a summarization assistant. "
    "Read the recent conversation messages and condense them into a brief, clear summary. "
    "Focus on the key facts, context, and questions that are relevant to answering the user. "
    "Avoid unnecessary details or repetitions. "
    "Format the summary in a structured way so that it can be easily consumed by another agent. "
    "Keep it short, factual, and precise."
    )
    # Inicializa el LLM
    llm = ChatOpenAI(
        model=model,
        reasoning_effort=reasoning_effort
    )
    middle = SummarizationMiddleware(
            model=llm,          # usa un modelo más ligero para resumir
            max_tokens_before_summary=max_tokens_before_summary,  # resume cuando pasa los 2000 tokens
            messages_to_keep=messages_to_keep, #mantiene solo los últimos 2 mensajes
            summary_prompt=summarization_prompt
            )
    
    return middle