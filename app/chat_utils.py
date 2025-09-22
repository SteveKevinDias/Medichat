from euriai.langchain import create_chat_model


#activate the LLM
def get_chat_model(api_key: str):
    return create_chat_model(api_key=api_key,
                             model = "gpt-4.1-nano",
                             temperature = 0.7)

#Getting the respose from the LLM
def ask_chat_model(chat_model,prompt: str):
    response = chat_model.invoke(prompt)
    return response.content

