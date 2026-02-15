from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
)

# Attempt 1:
# Observation: prompt1 > prompt4 > prompt2 > prompt3 > prompt5
prompt1 = "You are a factual assistant. Answer the question accurately and concisely. Use maximum of 3 words. Question: What is the capital of India? Answer:"
prompt2 = "You are a factual assistant. Answer the question accurately and concisely. Question: What is the capital of India? Answer:"
prompt3 = "Q: What is the capital of India?"
prompt4 = "Q: What is the capital of France A: Paris Q: What is the capital of India? A: "
prompt5 = "What is the capital of India?"

prompt6 = "The name of the capital of India is?"

# out = llm(prompt6)

# This for loop achieves the chatGPT-ish text generating tempo:
# for word in llm(prompt6, stream=True):
#    print(word, end="", flush=True)


# Attempt 2:
def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


question = "Which city is the capital of India?"

prompt = get_prompt(question)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
