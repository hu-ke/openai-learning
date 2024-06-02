import os
import tempfile
from langchain.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from sklearn.cluster import KMeans
import numpy as np
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma, Pinecone
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from extractImg import extract_cover
from dotenv import load_dotenv
load_dotenv()
BOOKS_DIR = Path() / 'books'
BOOKS_COVERS_DIR = Path() / 'book_covers'

app = FastAPI()
origins = [
    "http://localhost:5173",
    "https://reader.guru",
    "http://reader.guru"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv('OPENAI_API_KEY')

def load_book(file_obj, file_extension):
    """Load the content of a book based on its file type."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_obj.read())
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load()
            text = "".join(page.page_content for page in pages)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(temp_file.name)
            data = loader.load()
            text = "\n".join(element.page_content for element in data)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        os.remove(temp_file.name)
    text = text.replace('\t', ' ')
    return text

def split_and_embed(text, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    return docs, vectors


def cluster_embeddings(vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = [np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_]
    return sorted(closest_indices)


def summarize_chunks(docs, selected_indices, openai_api_key):
    llm3_turbo = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo-16k')
    map_prompt = """
    这里有一段来自一本书的段落。你的任务是对这段文字进行全面的总结。确保准确性，避免添加任何不在原文中的解释或额外细节。摘要至少应有三段，完整地表达出原文的要点。
    ```{text}```
    总结:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    selected_docs = [docs[i] for i in selected_indices]
    summary_list = []

    for doc in selected_docs:
        chunk_summary = load_summarize_chain(llm=llm3_turbo, chain_type="stuff", prompt=map_prompt_template).run([doc])
        summary_list.append(chunk_summary)
    
    return "\n".join(summary_list)

def create_final_summary(summaries, openai_api_key):
    llm4 = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=3000, model='gpt-4', request_timeout=120)
    combine_prompt = """
    这里有一些段落摘要，它们来自于一本书。你的任务是把这些摘要编织成一个连贯且详细的总结。读者应该能够从你的总结中理解书中的主要事件或要点。确保保持内容的准确性，并以清晰而引人入胜的方式呈现。
    ```{text}```
    综合总结:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template)
    final_summary = reduce_chain.run([Document(page_content=summaries)])
    return final_summary

def generate_summary(uploaded_file, openai_api_key, num_clusters=11, verbose=False):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = load_book(uploaded_file, file_extension)
    return text
    docs, vectors = split_and_embed(text, openai_api_key)
    selected_indices = cluster_embeddings(vectors, num_clusters)
    print(selected_indices)
    print(len(docs))
    print(len(vectors))
    summaries = summarize_chunks(docs, selected_indices, openai_api_key)
    final_summary = create_final_summary(summaries, openai_api_key)
    return final_summary

@app.post("/api/uploadfile/")
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    target_file = BOOKS_DIR / file_upload.filename
    cover_name = os.path.splitext(file_upload.filename)[0].lower()
    cover_name = f"{cover_name}.png"
    target_file_cover = BOOKS_COVERS_DIR / cover_name
    with open(target_file, 'wb') as f:
        f.write(data)
    
    extract_cover(target_file, target_file_cover)
    return {
        'code': 200,
        'msg': 'the book has been uploaded successfully.'
    }
    

@app.post("/api/summarize")
async def summarize_file(request: dict):
    print(request, request['filename'])
    save_to = BOOKS_DIR / request['filename']
    with open(save_to, 'rb') as file: 
        summary = generate_summary(file, openai_api_key, verbose=True)

    return {
        'code': 200,
        'msg': 'the summarization has been generated successfully.',
        'data': {
            'filename': request['filename'],
            'summary': summary
        }
    }

@app.get('/api')
async def root(): 
    return { 'message': 'Hello World'}

# Testing the summarizer
# if __name__ == '__main__':
#     openai_api_key = os.getenv('OPENAI_API_KEY')
#     # book_path = "./thethreekingdoms.pdf"
#     # book_path = './wentiejun.pdf'
#     book_path = './IntoThinAirBook.pdf'
#     with open(book_path, 'rb') as uploaded_file:
#         summary = generate_summary(uploaded_file, openai_api_key, verbose=True)
#         print(summary)