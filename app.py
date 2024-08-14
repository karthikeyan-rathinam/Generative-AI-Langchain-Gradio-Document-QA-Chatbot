import os
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from pypdf import PdfReader
import mimetypes
import validators
import requests
import tempfile
import gradio as gr
import openai


def get_empty_state():
    return {"knowledge_base": None}


def create_knowledge_base(docs):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=0, length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(chunks, embeddings)
    return knowledge_base


def upload_file(file_obj):
    try:
      loader = UnstructuredFileLoader(file_obj.name, strategy="fast")
      docs = loader.load()

      knowledge_base = create_knowledge_base(docs)
    except:
      text="Try Another file"
      return  file_obj.name, text

    return file_obj.name, {"knowledge_base": knowledge_base}


def upload_via_url(url):
    if validators.url(url):
        r = requests.get(url)

        if r.status_code != 200:
            raise ValueError(
                "Check the url of your file; returned status code %s" % r.status_code
            )

        content_type = r.headers.get("content-type")
        file_extension = mimetypes.guess_extension(content_type)
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
        temp_file.write(r.content)
        file_path = temp_file.name
        loader = UnstructuredFileLoader(file_path, strategy="fast",post_processors=[clean_extra_whitespace])
        docs = loader.load()
        knowledge_base = create_knowledge_base(docs)
        return file_path, {"knowledge_base": knowledge_base}
    else:
        raise ValueError("Please enter a valid URL")


def answer_question(question, state):

    try:
        knowledge_base = state["knowledge_base"]
        docs = knowledge_base.similarity_search(question)

        llm = OpenAI(temperature=0.4)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question)
        return response
    except:
        return "Please upload Proper Document"


with gr.Blocks(css="style.css",theme=gr.themes.Soft()) as demo:
    state = gr.State(get_empty_state())
    gr.HTML("""LINKAGETHINK""")
    with gr.Column(elem_id="col-container"):
        gr.HTML(
            """<hr style="border-top: 5px solid white;">"""
            )
        gr.HTML(
            """<br>
            <h1 style="text-align:center;">
               OPENAI Document QA
              </h1> """
        )
        gr.HTML(
            """<hr style="border-top: 5px solid white;">"""
            )

        gr.Markdown("**Upload your file**")
        with gr.Row(elem_id="row-flex"):
            with gr.Column(scale=0.85):
                file_url = gr.Textbox(
                    value="",
                    label="Upload your file",
                    placeholder="Enter a url",
                    show_label=False,
                    visible=False
                )
            with gr.Column(scale=0.90, min_width=160):
                file_output = gr.File(elem_classes="filenameshow")
            with gr.Column(scale=0.10, min_width=160):
                upload_button = gr.UploadButton(
                    "Browse File",file_types=[".txt", ".pdf", ".doc", ".docx",".json",".csv"],
                    elem_classes="filenameshow")
        with gr.Row():
          with gr.Column(scale=1, min_width=0):
            user_question = gr.Textbox(value="",label='Question Box :',show_label=True, placeholder="Ask a question about your file:",elem_classes="spaceH")
        with gr.Row():
          with gr.Column(scale=1, min_width=0):
            answer = gr.Textbox(value="",label='Answer Box :',show_label=True, placeholder="",lines=5)

    file_url.submit(upload_via_url, file_url, [file_output, state])
    upload_button.upload(upload_file, upload_button, [file_output,state])
    user_question.submit(answer_question, [user_question, state], [answer])

demo.queue().launch()
