from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.schema import TextNode
from llama_index.vector_stores.elasticsearch import AsyncDenseVectorStrategy
from llama_index.core import StorageContext, VectorStoreIndex
import glob
from llama_index.core import Settings
import json
import os
import openai
from dotenv import load_dotenv

load_dotenv('keys.env')
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


def print_results(results):
    for rank, result in enumerate(results, 1):
        print(
            f"{rank}.  score={result.get_score()} text={result.get_text()}"
        )

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data 

def create_index(nodes,vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index

def get_index(vector_store: ElasticsearchStore):
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

def search(index,query):
    print(">>> Documents:")
    retriever = index.as_retriever()
    results = retriever.retrieve(query,)
    print_results(results)
    print("\n>>> Answer:")
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
    return response

def create_textNodeList():
    all_files = glob.glob("json_files/*.json")
    json_list =[]
    for f in all_files:
        json_list.append(load_json(f))
    text_nodelist = []
    for j in json_list:
        doc_name = j['document_name']
        for item in j['document']:
            text =str(item['path'])+" "+ str(item['text_value'])
            node = TextNode(text=text,metadata={"document_name":doc_name,"page_number":item['text_page']})
            text_nodelist.append(node)
    return text_nodelist


def default_app(user_query):
    hybrid_store = ElasticsearchStore(
        es_api_key="Uk1ZVTVwRUJqY2hGaTZPX0kzTk46eVlmR202X0RRWE9MRjJGSF9Yal93UQ==",
        es_cloud_id="a37709bb28a646be9950834fc3f1e2e0:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ2NzZiY2ZiYTFkYmI0MzU5OGZlZjQ5OTI2ZTczMjliNiRlZjAwZDJiYWY1ZGQ0YmY0YTgwNzM2N2Q1YzE0MjUwYg==",
        index_name="default_index",
        retrieval_strategy=AsyncDenseVectorStrategy(hybrid=True),
        request_timeout=60
    )

    index = get_index(hybrid_store)
    search(index, user_query)
