from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from rag_main import *
from enum import Enum
from eval import *
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class ResponseModel(BaseModel):
    response: str
    sources: List[Dict[str, Any]]


class ResponseEval(BaseModel):
    method: str
    evaluation: Dict[str, Any]
    evaluation_file:str

class MethodEnum(str, Enum):
    method1 = "method1"
    method2 = "method2"
    method3 = "method3"
    method4 = "method4"
    method5 = "method5"



@app.post("/method1", response_model=ResponseModel, description = "This method first identifies which document to use for query and puts all the content in the LLM for answer generation")
async def call_method_1(query: QueryRequest):
    result = method_1_with_summary(query.query)
    return result

@app.post("/method2", response_model=ResponseModel,description = "This method first identifies which document to use for query and then retrieves the chunk using hybrid search and reranks them, the top 5 chunks are passed into llm ")
async def call_method_2(query: QueryRequest):
    result = method_2_with_reranking(query.query)
    return result

@app.post("/method3", response_model=ResponseModel, description = "This method first identifies which document to use for query and then retrieves the chunk using hybrid search applies RRF scoring, the top 5 chunks are passed into llm ")
async def call_method_3(query: QueryRequest):
    result = method_3_with_RRF(query.query)
    return result

@app.post("/method4", response_model=ResponseModel, description = "The chunking is changed here, earlier headings and subheadings used to be a part of the chunks.The same has been removed here.")
async def call_method_4(query: QueryRequest):
    result = method_4_without_heading_index(query.query)
    return result

@app.post("/method5", response_model=ResponseModel, description = "")
async def call_method_5(query: QueryRequest):
    result = method_5_without_doc_filter(query.query)
    return result

@app.post("/evaluate_method", response_model=ResponseEval, description="Evaluate different methods")
async def evaluate_methods(method: MethodEnum):
    method_mapping = {
        "method1": str(method_1_with_summary.__name__),
        "method2": str(method_2_with_reranking.__name__),
        "method3": str(method_3_with_RRF.__name__),
        "method4": str(method_4_without_heading_index.__name__),
        "method5": str(method_5_without_doc_filter.__name__)
    }
    evaluation_result = evaluate_methods_(method_mapping[method.value])  
    return evaluation_result
