# chainlit_app.py
# Implements:
# 1) "forget everything" -> clears memory WITHOUT calling the LLM
# 2) "use llama2" / "use orca" -> swaps model WITHOUT calling the LLM, keeps memory

import chainlit as cl
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate


# --- Model loader -------------------------------------------------------------
def load_llm(model_name: str) -> CTransformers:
    """
    WHY: We wrap model creation in a function so we can swap models on demand.
    - model_name is our own "switch" value: either "orca" or "llama2".
    - Returns a LangChain LLM wrapper (CTransformers) that can run GGUF locally.
    """
    if model_name == "llama2":
        return CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGUF",
            model_file="llama-2-7b-chat.Q2_K.gguf",
            model_type="llama2",
            max_new_tokens=200,  # cap response length so it doesn't ramble forever
        )

    # Default: Orca
    return CTransformers(
        model="zoltanctoth/orca_mini_3B-GGUF",
        model_file="orca-mini-3b.q4_0.gguf",
        model_type="llama2",
        max_new_tokens=200,
    )


# --- Prompt -------------------------------------------------------------------
PROMPT_TEMPLATE = """### System:
You are a helpful assistant. Answer clearly and concisely.

### Context:
{context}

### User:
{instruction}

### Response:
"""
# We keep a consistent "format" (System/Context/User/Response) so small models behave.
# {context} will come from ConversationBufferMemory.
# {instruction} is the new user message.

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "instruction"],
)
# LangChain needs to know which placeholders exist in the template.


# --- Session setup ------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    """
    WHY: This runs once per new chat session (per user tab/chat).
    Good place to:
    - set up memory
    - load default model (Orca)
    - create the chain and store it in the session
    """

    # 1) Create memory object
    memory = ConversationBufferMemory(memory_key="context")
    # This stores chat history under the key "context"
    # so it automatically fills {context} in our prompt.

    # 2) Load default model (Orca)
    llm = load_llm("orca")

    # 3) Create a chain that combines: prompt + llm + memory
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)

    # 4) Store both chain + "current model name" in session
    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("model_name", "orca")
    # We need to remember what model is active when user asks to switch.

    await cl.Message("Orca loaded. Ask me anything.").send()


# --- Helpers for memory + model switching ------------------------------------
def clear_memory(llm_chain: LLMChain) -> None:
    """
    For 'forget everything' we must reset conversational memory.
    ConversationBufferMemory stores history in memory.chat_memory.
    """
    llm_chain.memory.clear()  # resets the memory buffer


def swap_model(llm_chain: LLMChain, model_name: str) -> None:
    """
    'use llama2' / 'use orca' must replace the model *without* losing history.
    We keep the same llm_chain + same memory object, only replace llm_chain.llm.
    """
    llm_chain.llm = load_llm(model_name)


# --- Main message handler -----------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    """
    Runs every time the user sends a message.
    We intercept special commands BEFORE calling the LLM.
    """

    text = message.content.strip()
    text_lower = text.lower()

    # Get the chain from session
    llm_chain: LLMChain = cl.user_session.get("llm_chain")
    # This is the object that holds:
    # - the LLM
    # - the prompt
    # - the memory

    # --- (1) Forget command ---------------------------------------------------
    if text_lower == "forget everything":
        clear_memory(llm_chain)  # reset memory buffer
        await cl.Message("Uh oh, I've just forgotten our conversation history").send()
        return  # stop here so the LLM is never called

    # --- (2) Model switching commands ----------------------------------------
    if text_lower == "use llama2":
        swap_model(llm_chain, "llama2")
        cl.user_session.set("model_name", "llama2")
        await cl.Message("Model changed to Llama").send()
        return

    if text_lower == "use orca":
        swap_model(llm_chain, "orca")
        cl.user_session.set("model_name", "orca")
        await cl.Message("Model changed to Orca").send()
        return

    # --- Normal user message: call the LLM -----------------------------------

    msg = cl.Message(content="")
    await msg.send()

    # LLMChain expects a dict matching prompt variables.
    # Here {instruction} is the user message. {context} is supplied by memory automatically.
    full_response = ""

    # Chainlit provides a built-in LangChain streaming callback handler.
    # It streams tokens nicely to the UI.
    res = await llm_chain.ainvoke(
        {"instruction": text},
        config={"callbacks": [cl.AsyncLangchainCallbackHandler()]},
    )

    # res is typically a dict like {"text": "..."} depending on LangChain version.
    # We also finalize the UI message ourselves, just in case.
    if isinstance(res, dict) and "text" in res:
        full_response = res["text"]
    else:
        # fallback: best-effort string conversion
        full_response = str(res)

    # Send final message content (if callback handler already streamed, this will still be fine)
    await cl.Message(full_response).send()
