import streamlit as st 
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)

def init_page() -> None:
  st.set_page_config(
    page_title="LLM_Bot"
  )
  st.header("LLM_Bot")
  st.sidebar.title("Options")

def init_messages() -> None:
  clear_button = st.sidebar.button("Clear Conversation", key="clear")
  if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
      SystemMessage(
        content="you are a helpful AI assistant. Reply your answer in markdown format."
      )
    ]
    
# chat application with history 
def format_chat_prompt(message, history, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    chat = []
    # Convert all messages in history to chat interactions
    for interaction in history:
        chat.append({"role": "user", "content" : interaction[0]})
        chat.append({"role": "assistant", "content" : interaction[1]})
    # Add the new message
    chat.append({"role": "user", "content" : message})
    # Generate the prompt, verifying that we don't go beyond the maximum number of tokens
    for i in range(0, len(chat), 2):
        # Generate candidate prompt with the last n-i entries
        prompt = tokenizer.apply_chat_template(chat[i:], tokenize=False)
        # Tokenize to check if we're over the limit
        tokens = tokenizer(prompt)
        if len(tokens.input_ids) <= max_tokens:
            # We're good, stop here
            return prompt
    # We shall never reach this line
    raise SystemError

def chat(message, history, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForCausalLM.from_pretrained("distilbert-base-uncased")
    prompt = format_chat_prompt(message, history, max_tokens)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs,
                             max_new_tokens=128)
    outputs = outputs[0, inputs.input_ids.size(-1):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    history.append([message, response])
    # string returned 
    return response 

def get_answer(messages) -> str: 
  history = [] 
  max_tokens = 1024 
  response = chat(messages, history, max_tokens) 
  return response 

def main() -> None:
  init_page()
  init_messages()

  if user_input := st.chat_input("Input your question!"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("Bot is typing ..."):
      answer = get_answer(user_input)
      print(answer)
    st.session_state.messages.append(AIMessage(content=answer))
    

  messages = st.session_state.get("messages", [])
  for message in messages:
    if isinstance(message, AIMessage):
      with st.chat_message("assistant"):
        st.markdown(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("user"):
        st.markdown(message.content)

if __name__ == "__main__":
  main()
