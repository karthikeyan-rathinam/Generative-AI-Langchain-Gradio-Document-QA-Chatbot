
# The Generative AI Langchain Document QA Chatbot:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)![Kali](https://img.shields.io/badge/Kali-268BEE?style=for-the-badge&logo=kalilinux&logoColor=white)![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white)![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)![Google](https://img.shields.io/badge/google-4285F4?style=for-the-badge&logo=google&logoColor=white)![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo-DE5833?style=for-the-badge&logo=DuckDuckGo&logoColor=white)![Edge](https://img.shields.io/badge/Microsoft%20Edge-0078D7.svg?style=for-the-badge&logo=Microsoft-Edge&logoColor=white)![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)![Open Project](https://img.shields.io/badge/OpenProject-0770B8.svg?style=for-the-badge&logo=OpenProject&logoColor=white)![Open Access](https://img.shields.io/badge/Open%20Access-F68212.svg?style=for-the-badge&logo=Open-Access&logoColor=white)

# **Let's connect :**

[![GitHub](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/karthikeyanrathinam/)
[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/karthikeyan-rathinam/)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/@linkagethink)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335.svg?style=for-the-badge&logo=Gmail&logoColor=white)](mailto:karthikeyanr1801@gmail.com)

Here is an explanation of the OpenAI Langchain Document QA code:

## The main components:

- `UnstructuredFileLoader` - Loads documents from various file types (txt, pdf, docx, etc) into a list of text documents that Langchain can process.

- `OpenAIEmbeddings` - Creates vector embeddings for each chunk of text using OpenAI's embedding model. This encodes the semantic meaning of each chunk.

- `FAISS` - Indexes the vector embeddings to create the actual searchable knowledge base. This allows fast similarity search over the embeddings/documents.

- `CharacterTextSplitter` - Splits the documents into smaller chunks before embedding. This allows embedding documents of arbitrary size.

- `load_qa_chain` - Loads a pretrained QA model from Langchain using the OpenAI LLM. This is a T5 model fine-tuned for question answering.

## The main functions:

- `create_knowledge_base` - Takes documents, splits them, embeds them and indexes into a knowledge base.

- `upload_file` - Handler for uploading a local file. Loads it and creates the knowledge base.

- `upload_via_url` - Handles a URL, downloads the content, and loads it.

- `answer_question` - Takes a question and searches the knowledge base embeddings to find the most relevant passage. Passes this to the QA model to generate an answer.

## The Gradio app:

- Allows uploading local files or entering URLs to populate the knowledge base.

- Provides a textbox to enter a question.

- Displays the generated answer returned by the QA model.

# Loading documents

The `UnstructuredFileLoader` handles ingesting files of different formats like PDF, DOC, TXT etc.

```python
loader = UnstructuredFileLoader(file_path, strategy="fast")
docs = loader.load()
```

It uses heuristic strategies to extract text content from these file types.

The resulting `docs` is a list of text snippets representing the contents of the uploaded file.

# Text Splitting

The `CharacterTextSplitter` splits the loaded documents into smaller chunks:

```python
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)
```

This is done because large documents can't be embedded efficiently in one go. The chunks will be embedded separately.

# Embedding

The `OpenAIEmbeddings` module is used to generate vector embeddings for each chunk:

```python
embeddings = OpenAIEmbeddings()
```

This uses OpenAI's text-embedding-ada-002 model under the hood.

# Indexing

The vector embeddings for each chunk are indexed using FAISS for fast similarity search:

```python
knowledge_base = FAISS.from_documents(chunks, embeddings)
```

This indexing allows finding the most relevant chunks for a query.

# Question Answering

Given a user question, relevant chunks are retrieved by searching the knowledge base index:

```python
docs = knowledge_base.similarity_search(question)
```

These chunks are passed to a QA model to generate the answer:

```python
llm = OpenAI()
chain = load_qa_chain(llm)
response = chain.run(docs, question)
```

The `load_qa_chain` helper sets up the QA model using the OpenAI API.

So in summary, it provides an end-to-end pipeline for uploading documents, indexing them for search, and leveraging a QA model to answer questions based on the uploaded doc contents.

# **Follow**
Feel free to reach out if you have any questions or need further assistance.


[![GitHub](https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/karthikeyanrathinam/)
[![Linkedin](https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white)](https://www.linkedin.com/in/karthikeyan-rathinam/)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/@linkagethink)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335.svg?style=for-the-badge&logo=Gmail&logoColor=white)](mailto:karthikeyanr1801@gmail.com)
