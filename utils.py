from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np
import json
import os
import openai
from dotenv import load_dotenv
import glob
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from FlagEmbedding import FlagReranker



load_dotenv('keys.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

class CustomES:
    def __init__(self, index:str=None,rerank:bool = False) -> None:
        self.__es_api_key = os.getenv("ES_API_KEY")
        self.__es_cloud_id = os.getenv("ES_CLOUD_ID")
        self.rerank = rerank
        self.es = Elasticsearch(api_key = self.__es_api_key ,cloud_id = self.__es_cloud_id, timeout=60)
        self.index_name = index
        self.dim = 1024
        self.__doc_filter =False
        self.bulk_data = []
        self.text_mapping = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "doc_name": {
                "type": "keyword" 
            },
                    "page_num": {"type": "integer"},
                    "embedding": {"type": "dense_vector", "dims": self.dim}
                }
            }
        }
        self.search_key = "text"
        self.summary_mapping = {
            "mappings": {
                "properties": {
                    "summary": {"type": "text"},
                    "doc_name": {
                "type": "keyword"
            },
                    "doc_text": {"type": "text"},
                }
            }
        }
    
    def initialize_model(self):
        self.model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",trust_remote_code=True)
        
    def __bulk_update(self,bulk_data):
        if bulk_data:
            response = bulk(self.es, bulk_data)
            print(f"Bulk indexing response: {response}")
        else:
            print("No documents to index.")

    def create_text_index(self, index_name, text_list, page_list, doc_name_list):
        self.index_name = index_name
        if not hasattr(self, 'model'):
            self.initialize_model()
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=self.text_mapping)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists, adding new documents.")
        bulk_data = []
        ### embedding to list of embedding
        embedding = self.model.encode(text_list).tolist()
        print(len(embedding))  


        for  text, page_number, doc_name, emb in zip(text_list, page_list, doc_name_list,embedding):
            doc = {
                "_index": index_name,
                "_source": {
                    "text": text,
                    "doc_name": doc_name ,
                    "page_num": page_number,
                    "embedding": emb
                }
            }
            bulk_data.append(doc)
        self.__bulk_update(bulk_data)
    
    def create_summary_index(self, index_name, summary_list, doc_text_list, doc_name_list):
        self.index_name = index_name
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=self.summary_mapping )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists, adding new documents.")
        bulk_data = []
        for  summary, doc_name, doc_text in zip(summary_list, doc_name_list, doc_text_list):
            doc = {
                "_index": index_name,
                "_source": {
                    "summary": summary,
                    "doc_name": doc_name ,
                    "doc_text": doc_text
                }
            }
            bulk_data.append(doc)
        self.__bulk_update(bulk_data)
    
    def get_all_index_names(self):
        self.__all_indices = self.es.indices.get_alias(index="*").keys()
        custom_indices = [index for index in self.__all_indices if not index.startswith('.')]
        return  custom_indices

    def delete_index(self, index_name):
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            print(f"Index '{index_name}' deleted successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")
        
    def bm25_search(self,top_k,query):
        self.query = query
        if self.__doc_filter ==False:
            bm25_result = self.es.search(index=self.index_name, body={
                "query": {
                    "match": {
                        self.search_key: self.query
                    }
                },
                "size": top_k
            })
        else:
            val  ={
                "query": {
                    "match": {
                        self.search_key: self.query
                    },
                    "terms": self.doc_query
                },
                "size": top_k
            }
            print("query:", val)
            bm25_result = self.es.search(index=self.index_name, body={
                "query": {
                    "bool": {
                        "must": [
                            {"match": {self.search_key: self.query}}
                        ],
                        "filter": [
                            {"terms": self.doc_query}
                        ]
                    }
                },
                "size": top_k
            })
            
        return bm25_result
    
    def knn_search(self,top_k,query):
        self.query = query
        if not hasattr(self, 'model'):
            self.initialize_model()
        query_embedding = self.model.encode([self.query])[0].tolist()
        print(len(query_embedding))
        if self.__doc_filter==False:
            knn_result = self.es.search(index=self.index_name, body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k
                }
            })
        else:
            knn_result = self.es.search(index=self.index_name, body={
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_embedding,
                        "k": top_k
                    },
                    "query": {
                            "bool": {
                            "filter": [
                                {
                                "terms": self.doc_query
                                }
                            ]
                            }
                        }
                })
            
        return knn_result
    
    def cross_enc_reranking(self, query, retrieved_texts):
        """
        Use cross-encoder for re-ranking question-answer pairs.
        """
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        pairs = [[query, text] for text in retrieved_texts]
        scores = reranker.compute_score(pairs)
        return scores
    
    def hybrid_search(self,query, key="text",top_k=5, k1 = 60):
        if not hasattr(self, 'model'):
            self.initialize_model()
        self.search_key = key
        self.query = query
        print("query is :", query)
        bm25_result = self.bm25_search(top_k,query)
        knn_result = self.knn_search(top_k,query)
        if self.rerank==False:
            ### reciprocal rank fusion

            self.rrf_k1 = k1
            bm25_hits = {hit['_id']: (rank, hit['_score']) for rank, hit in enumerate(bm25_result['hits']['hits'])}
            knn_hits = {hit['_id']: (rank, hit['_score']) for rank, hit in enumerate(knn_result['hits']['hits'])}
            
            combined_scores = {}
            for doc_id, (bm25_rank, _) in bm25_hits.items():
                bm25_rrf_score = 1 / (self.rrf_k1 + bm25_rank)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + bm25_rrf_score
            
            for doc_id, (knn_rank, _) in knn_hits.items():
                knn_rrf_score = 1 / (self.rrf_k1 + knn_rank)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + knn_rrf_score
            
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_results = []
            for doc_id, rrf_score in sorted_results[:top_k]:
                doc_source = self.es.get(index=self.index_name, id=doc_id)['_source']
                top_results.append({
                    "id": doc_id,
                    self.search_key: doc_source[self.search_key], 
                    "page_num":doc_source["page_num"],
                    "doc_name":doc_source["doc_name"],
                    "reciprocal_rank_fusion_scores":rrf_score
                })
        else:
            bm25_hits = {hit['_id']: hit['_source'] for hit in bm25_result['hits']['hits']}
            knn_hits = {hit['_id']: hit['_source'] for hit in knn_result['hits']['hits']}
            combined_hits = {**bm25_hits, **knn_hits}  
            retrieved_texts = [hit[self.search_key] for hit in combined_hits.values()]
            re_rank_scores = self.cross_enc_reranking(query, retrieved_texts)
            ranked_results = sorted(zip(combined_hits.keys(), re_rank_scores), key=lambda x: x[1], reverse=True)
            top_results = []
            for doc_id, score in ranked_results[:top_k]:
                doc_source = self.es.get(index=self.index_name, id=doc_id)['_source']
                top_results.append({
                    "id": doc_id,
                    self.search_key: doc_source[self.search_key],
                    "page_num":doc_source["page_num"],
                    "doc_name":doc_source["doc_name"],
                    "re_rank_score": score
                })
        return top_results
        
    def filter_text_index(self, filter_list_docs:list):
        """implemented for querying in docs only"""
        self.__doc_filter = True
        self.doc_query = {
                    "doc_name": filter_list_docs
                }
    
    def filter_summary_index(self, filter_list_docs:list):
        print(type(filter_list_docs))
        print(self.index_name)
        mapping = self.es.indices.get_mapping(index=self.index_name)
        print(mapping)
        filter_result = self.es.search(index=self.index_name, body = 
                    {
            "query": {
                "terms": {
                "doc_name": filter_list_docs
                }
            }
            }
        )
        # print(filter_result)
        
        # if filter_result["hits"]['total']['value']==0:
            # print(self.retrieve_summary_index())
        res= []
        for f in filter_result['hits']['hits']:
            res.append(f['_source'])
        return res
        
    
    def retrieve_summary_index(self,index_name ="text_summary_index" ):
        """
        Retrieve all fields in the specified Elasticsearch index.
        """
        self.index_name = index_name
        if not self.es.indices.exists(index=index_name):
            print(f"Index '{index_name}' does not exist.")
            return None
        response = self.es.search(
            index=index_name,
            body={
                "query": {
                    "match_all": {}
                },
                "_source": True, 
            }
        )
        
        documents = response['hits']['hits']
        all_docs = []
        for doc in documents:
            doc_fields = doc['_source'] 
            all_docs.append(doc_fields)
        return all_docs

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data 
    
def get_textNodeList_with_headers():
    ## todo: handle empty files input
    all_files = glob.glob("json_files/*.json")
    json_list =[]
    for f in all_files:
        json_list.append(load_json(f))
    text_list = []
    page_list = []
    doc_namelist = []
    for j in json_list:
        doc_name = j['document_name']
        for item in j['document']:
            text =str(item['path'])+" "+ str(item['text_value'])
            text_list.append(text)
            page_list.append(item['text_page'])
            doc_namelist.append(doc_name)
    return text_list,page_list,doc_namelist

def get_textNodeList():
    ## todo: handle empty files input
    all_files = glob.glob("json_files/*.json")
    json_list =[]
    for f in all_files:
        json_list.append(load_json(f))
    text_list = []
    page_list = []
    doc_namelist = []
    for j in json_list:
        doc_name = j['document_name']
        for item in j['document']:
            text = str(item['text_value'])
            text_list.append(text)
            page_list.append(item['text_page'])
            doc_namelist.append(doc_name)
    return text_list,page_list,doc_namelist

def get_summaryNodeList():
    all_files = glob.glob("summary_json/*.json")
    json_list =[]
    for f in all_files:
        json_list.append(load_json(f))
    text_list = []
    summmary_list = []
    doc_namelist = []
    for j in json_list:
            text =str(j['doc_text'])
            summary = str(j['doc_summary'])
            text_list.append(text)
            summmary_list.append(summary)
            doc_namelist.append(j["doc_name"])
    return text_list,summmary_list,doc_namelist
        



### method1 direct hybrid search all results
### method2 direct hybrid search with re-ranking
### method3 hybrid search with doc filter
### method3 answering from a doc, 
### method4 