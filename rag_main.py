###
from utils import *
import logging
import sys
import os
import json
import glob
from openai import OpenAI
from llama_index.core import PromptTemplate

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = OPENAI_API_KEY)


text_index = "text_doc_index"
summary_index = "text_summary_index"

class RAG():
    def __init__(self) -> None:
        self.text_index = "text_doc_index"
        self.summary_index = "text_summary_index"
        self.sys_prompt = "You are a smart Assistant who can obtain the outputs by following the given intructions"

    def llm_response(self,prompt, sys_prompt,temp=0.2, model_name="gpt-4o-mini", json_format= True):
        if json_format:
            answer = client.chat.completions.create(
                model=model_name,
                messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ],
                temperature=temp,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={ "type": "json_object" },
                n=1,
            )
        else:
            answer = client.chat.completions.create(
                model=model_name,
                messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
                ],
                temperature=temp,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
            )
        return answer.choices[0].message.content
    
    def initialize_summary_dict(self):
        self.summary_es = CustomES(index= self.summary_index)
        self.summary_dict = self.summary_es.retrieve_summary_index()
        self.summary_list = [doc['summary'] for doc in self.summary_dict]
        

    def get_question_type(self,query):
        "todo format summary_list"
        if not hasattr(self, 'summary_dict'):
            self.initialize_summary_dict()
        formatted_strings = [f"- {doc['summary']}" for doc in self.summary_dict]
        context_str = "\n\n".join(formatted_strings)
        ques_type_template_str=f"""
    The context of the documents is provided below:
    ---------------------
    {context_str}
    ---------------------
    You are given a query. Based on the provided context, identify the type of query from the following categories: [summary, specific, general]

    Descriptions for each category:
    summary: The query is requesting a summary/ brief explaination or requesting very descriptiv answer related to the provided context.
    specific: The query seeks specific information from the provided context.
    general: The query is unrelated to the provided context.

    Return the Answer in JSON with keys as "category"

    Query: {query} 
    Answer:  
    """
        response = self.llm_response(ques_type_template_str,self.sys_prompt,model_name="gpt-4o")
        response_json = json.loads(response)
        print(response_json)
        return response_json

    def get_document_name(self,query):
        sys_prompt = "You are an intelligent AI assistant designed to provide answers related to Human Resource policies for the company Simpplr."
        if not hasattr(self, 'summary_dict'):
            self.initialize_summary_dict()
        formatted_strings = [f"- {doc['doc_name']} : {doc['summary']}" for doc in self.summary_dict]
        context_str = "\n\n".join(formatted_strings)
        doc_name_list = [doc['doc_name'] for doc in self.summary_dict]
        doc_type_templat_str = f"""
Below are the summaries of each document, with each document name and its corresponding summary separated by a delimiter (:). The document name is listed first, followed by the summary.
---------------------
{context_str}
---------------------
Based on the given summaries and the query, determine which documents are relevant for answering the query. Ensure the document names in your response match exactly with one of the names from the following list: {doc_name_list}.
Return the document names in JSON format with the key as "document" and the value as a Python list containing the relevant document names. If no documents are suitable, respond with "Don't Know" as the value.

Question: {query} 
Document Name:
"""
        response = self.llm_response(doc_type_templat_str,sys_prompt, model_name="gpt-4o")
        response_json = json.loads(response)
        print(response_json)
        
        return response_json

    def generate_answer(self,query, contexts):
        sys_prompt = "You are an intelligent AI assistant designed to provide answers related to Human Resource policies for the company Simpplr."

        formatted_strings = [f"- {doc}" for doc in contexts]
        context_str = "\n\n".join(formatted_strings)
        rag_template_str = f"""
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: 
"""
        response = self.llm_response(rag_template_str,sys_prompt, json_format=False )
        print(response)
        return response
    
    def general_response(self,query):
        sys_prompt = "You are an intelligent AI assistant designed to provide answers related to Human Resource policies for the company Simpplr."
        rag_template_str = f"""
        Prompt the user to give queries related to the Human resource policy documents of Simpplr organization. If the question is irrelevant to the policies give answer as dont know or apologize that you can only provide answers relevant to queries in HR policy documents.
Query: {query}
Answer: 
"""
        response = self.llm_response(rag_template_str,sys_prompt, json_format=False )
        return response

def method_1_with_summary(user_query):
    structured_ans = {
        "response":"",
        "sources" : [],
    }
    ques_type_list = ["summary", "specific", "general"]
    """doc type detection using summmary and qa using the doc_text"""
    answer_generation = RAG()
    ques_type = answer_generation.get_question_type(user_query)['category']
    docs_list = answer_generation.get_document_name(user_query)['document']
    # print(docs_list[0].lower())
    if len(docs_list)==1 and "know" in docs_list[0].lower():
        answer = answer_generation.general_response(user_query)
        sources = []
    else:
        if ques_type.lower() in ques_type_list:
            summary_dict_list = answer_generation.summary_es.filter_summary_index(docs_list)
            sources = []
            print(docs_list)
            if ques_type.lower()=="summary":
                summary_list = [s["summary"] for s in summary_dict_list]
                for s in summary_dict_list:
                    new_dict = {}
                    new_dict['text'] = s['summary']
                    new_dict['document_name'] = s['doc_name']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,summary_list)
            elif ques_type.lower()=="specific":
                txt_list = []
                for s in summary_dict_list:
                        new_dict  ={}
                        txt_list.append(s['doc_text'])
                        new_dict['text'] = s['doc_text']
                        new_dict['document_name'] =s['doc_name']
                        sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,txt_list)
            else:
                answer = answer_generation.general_response(user_query)
                
    structured_ans['response'] = answer
    structured_ans['sources'] = sources
    return structured_ans

def method_2_with_reranking(user_query):
    text_index = "text_doc_index"
    structured_ans = {
        "response":"",
        "sources" : [],
    }
    ques_type_list = ["summary", "specific", "general"]
    answer_generation = RAG()
    ReRankRetriever = CustomES(index=text_index,rerank=True)
    ques_type = answer_generation.get_question_type(user_query)['category']
    docs_list = answer_generation.get_document_name(user_query)['document']
    # print(docs_list[0].lower())
    if len(docs_list)==1 and "know" in docs_list[0].lower():
        answer = answer_generation.general_response(user_query)
        sources = []
    else:
        if ques_type.lower() in ques_type_list:
            sources = []
            print(docs_list)
            if ques_type.lower()=="summary":
                summary_dict_list = answer_generation.summary_es.filter_summary_index(docs_list)
                summary_list = [s["summary"] for s in summary_dict_list]
                for s in summary_dict_list:
                    new_dict = {}
                    new_dict['text'] = s['summary']
                    new_dict['document_name'] = s['doc_name']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,summary_list)
            elif ques_type.lower()=="specific":
                ReRankRetriever.filter_text_index(docs_list)
                retrieved_docs = ReRankRetriever.hybrid_search(user_query)
                txt_list = []
                for s in retrieved_docs:
                    new_dict = {}
                    txt_list.append(s['text'])
                    new_dict['text'] = s['text']
                    new_dict['document_name'] =s['doc_name']
                    new_dict['page'] = s['page_num']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,txt_list)
            else:
                answer = answer_generation.general_response(user_query)
                
    structured_ans['response'] = answer
    structured_ans['sources'] = sources
    return structured_ans

def method_3_with_RRF(user_query):
    """reciprocal rank fusion"""
    text_index = "text_doc_index"
    structured_ans = {
        "response":"",
        "sources" : [],
    }
    ques_type_list = ["summary", "specific", "general"]
    answer_generation = RAG()
    RRFRetriever = CustomES(index=text_index)
    ques_type = answer_generation.get_question_type(user_query)['category']
    docs_list = answer_generation.get_document_name(user_query)['document']
    # print(docs_list[0].lower())
    if len(docs_list)==1 and "know" in docs_list[0].lower():
        answer = answer_generation.general_response(user_query)
        sources = []
    else:
        if ques_type.lower() in ques_type_list:
            sources = []
            print(docs_list)
            if ques_type.lower()=="summary":
                summary_dict_list = answer_generation.summary_es.filter_summary_index(docs_list)
                summary_list = [s["summary"] for s in summary_dict_list]
                for s in summary_dict_list:
                    new_dict = {}
                    new_dict['text'] = s['summary']
                    new_dict['document_name'] = s['doc_name']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,summary_list)
            elif ques_type.lower()=="specific":
                RRFRetriever.filter_text_index(docs_list)
                retrieved_docs = RRFRetriever.hybrid_search(user_query)
                txt_list = []
                for s in retrieved_docs:
                    new_dict = {}
                    txt_list.append(s['text'])
                    new_dict['text'] = s['text']
                    new_dict['document_name'] =s['doc_name']
                    new_dict['page'] = s['page_num']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,txt_list)
            else:
                answer = answer_generation.general_response(user_query)
                
    structured_ans['response'] = answer
    structured_ans['sources'] = sources
    return structured_ans

def method_4_without_heading_index(user_query):
    """reciprocal rank fusion"""
    text_index = "text_doc_without_headings_index"
    structured_ans = {
        "response":"",
        "sources" : [],
    }
    ques_type_list = ["summary", "specific", "general"]
    answer_generation = RAG()
    RRFRetriever = CustomES(index=text_index)
    ques_type = answer_generation.get_question_type(user_query)['category']
    docs_list = answer_generation.get_document_name(user_query)['document']
    # print(docs_list[0].lower())
    if len(docs_list)==1 and "know" in docs_list[0].lower():
        answer = answer_generation.general_response(user_query)
        sources = []
    else:
        if ques_type.lower() in ques_type_list:
            sources = []
            print(docs_list)
            if ques_type.lower()=="summary":
                summary_dict_list = answer_generation.summary_es.filter_summary_index(docs_list)
                summary_list = [s["summary"] for s in summary_dict_list]
                for s in summary_dict_list:
                    new_dict = {}
                    new_dict['text'] = s['summary']
                    new_dict['document_name'] = s['doc_name']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,summary_list)
            elif ques_type.lower()=="specific":
                RRFRetriever.filter_text_index(docs_list)
                retrieved_docs = RRFRetriever.hybrid_search(user_query)
                txt_list = []
                for s in retrieved_docs:
                    new_dict = {}
                    txt_list.append(s['text'])
                    new_dict['text'] = s['text']
                    new_dict['document_name'] =s['doc_name']
                    new_dict['page'] = s['page_num']
                    sources.append(new_dict)
                answer = answer_generation.generate_answer(user_query,txt_list)
            else:
                answer = answer_generation.general_response(user_query)
                
    structured_ans['response'] = answer
    structured_ans['sources'] = sources
    return structured_ans

def method_5_without_doc_filter(user_query):
    text_index = "text_doc_index"
    structured_ans = {
        "response":"",
        "sources" : [],
    }
    ques_type_list = ["summary", "specific", "general"]
    """doc type detection using summmary and qa using the doc_text"""
    answer_generation = RAG()
    ReRankRetriever = CustomES(index=text_index,rerank=True)
    ques_type = answer_generation.get_question_type(user_query)['category']
    # print(docs_list[0].lower())
    # if len(docs_list)==1 and "know" in docs_list[0].lower():
    #     answer = answer_generation.general_response(user_query)
    #     sources = []
    # else:
    if ques_type.lower() in ques_type_list:
        sources = []
        if ques_type.lower()=="summary":
            answer_generation.summary_es.search_key = "summary"
            summary_dict_list = answer_generation.summary_es.bm25_search(5,user_query)
            summary_list = [s["summary"] for s in summary_dict_list]
            for s in summary_dict_list:
                new_dict = {}
                new_dict['text'] = s['summary']
                new_dict['document_name'] = s['doc_name']
                sources.append(new_dict)
            answer = answer_generation.generate_answer(user_query,summary_list)
        elif ques_type.lower()=="specific":
            retrieved_docs = ReRankRetriever.hybrid_search(user_query)
            txt_list = []
            for s in retrieved_docs:
                new_dict = {}
                txt_list.append(s['text'])
                new_dict['text'] = s['text']
                new_dict['document_name'] =s['doc_name']
                new_dict['page'] = s['page_num']
                sources.append(new_dict)
            answer = answer_generation.generate_answer(user_query,txt_list)
        else:
            answer = answer_generation.general_response(user_query)
            
    structured_ans['response'] = answer
    structured_ans['sources'] = sources
    return structured_ans

def get_results_syntheticData(method):
    file_path = "eval_data/20240916_043352.json"
    with open(file_path, 'r') as f:
        data = json.load(f)  
    for entry in data:
        question = entry.get("input", None)
        if not question:
            continue  # If no input is found, skip to the next
        method_response = method(question)
        answer = method_response.get("response", None)
        retrieval_context = []
        sources = method_response.get("sources", None)
        if sources:
            for s in sources:
                retrieval_context.append(s.get("text", None))
        entry["actual_output"] = answer
        entry["retrieval_context"] = retrieval_context
    outfile = f"./eval_methods/{str(method.__name__)}.json"
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)
    return data  # Return the updated data (optional)

# get_results_syntheticData(method_4_without_heading_index)
# get_results_syntheticData(method_2_with_reranking)
# get_results_syntheticData(method_3_with_RRF)

# print(method_4_without_heading_index("Briefly, explain me the leave policy content?"))
# print(method_4_without_heading_index("How many leaves can I take?"))
# print(method_4_without_heading_index("Explain the ESOP policy to me?"))
# print(method_4_without_heading_index("Tell me about russia ukraine war?"))

# print(method_1("Hey, I am Nayan"))

# [cls] question [sep] contxt1 -> encoder-> + Ve







"""wrap all the methods in rest apis"""
"""
detect question type
 summary type
 specific question
 multiple queries in one
 general hi/hello

query rewriting:
    splitting the queries into 2 separate queries or a standalone query. 
"""

"""
method 1.
pass question and obtain which doc will be able to answer the query.
for summary pass only the summaries of the docs
pass the doc content and the question and generate answer from it using llm
"""

# prompt1 = 


"""
method 2.
pass question and obtain which doc will be able to answer the query.
filter the doc and retrieve using the question,hybrid search and re ranking
"""

"""
method 3.
pass question and all the documents text data 
"""

"""
method 4.
plain knn search on docs, without any summarization
"""






