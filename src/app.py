import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from Pipeline import ChatPipeline

pipeline = ChatPipeline(max_history=10)


def chat(message: str, history: list[dict]) -> str:
    return pipeline.chat(message)


def on_clear():
    pipeline.reset_session()


with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="Human - AI(아이)",
        description="대화 내용을 Graph DB에 기억하는 프로토타입 챗봇",
    )
    chatbot.chatbot.clear(on_clear)
    demo.unload(lambda: pipeline.close_session())

if __name__ == "__main__":
    demo.launch()
