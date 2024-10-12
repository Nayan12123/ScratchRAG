from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
import nest_asyncio 
from openai import OpenAI
nest_asyncio.apply()
import json
import glob
from collections import defaultdict

def get_llm_JSONresponse(prompt, sys_prompt,temp=0.2, model_name="gpt-4o-mini"):
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
    return answer.choices[0].message.content

def find_all_key_paths(data, path=None):
    if path is None:
        path = []
    
    all_paths = []
    
    for key, value in data.items():
        current_path = path + [key]
        all_paths.append(current_path)
        
        if isinstance(value, dict):
            all_paths.extend(find_all_key_paths(value, current_path))
    
    return all_paths      

def get_document_structure(json_list):
    heading_list = []
    for item_list in json_list:
        item_list = item_list['items']
        for i in item_list:
            if i['type']=="heading":
                heading_list.append(i['value'])
    heading_count = defaultdict(int) 
    unique_heading_list = []

    for heading in heading_list:
        heading_count[heading] += 1
        if heading_count[heading] > 1:  
            unique_heading_list.append(f"{heading} |ID:{heading_count[heading]}")
        else:
            unique_heading_list.append(heading)
    sys_prompt = """You are a smart Assistant who can follow and obtain the outputs by following the given intructions"""
    prompt = f"""
You are given a python list of headings obtained from the document, obtain the document structure in json format using the headings list. 
Strictly ensure that the spellings of any elements in the Headings List should not be changed in output document structure. You can remove some headings if they dont't fit in the document structure.
The final output should contain keys as one entry from heading list and the values should be a dictionary consisting of subheadings if any else can be an empty dictionary and same will be repeated across the whole json output. 
Headings List: {unique_heading_list}
Here is the final structure of the document:
    """
    headings_structure = get_llm_JSONresponse(prompt, sys_prompt)
    headings_structure = json.loads(headings_structure)
    print("llm output")
    print(headings_structure)
    return headings_structure

def extract_keys(json_obj, parent_key=''):
    keys_list = []
    for k, v in json_obj.items():
        full_key = f"{parent_key}{k}" if parent_key == '' else f"{parent_key} > {k}"
        keys_list.append(full_key)
        if isinstance(v, dict) and v:
            keys_list.extend(extract_keys(v, full_key + ' > '))
    return keys_list

def extract_text_and_heading(new_list):
    result = []
    last_headings = []  
    for obj in new_list:
        if obj.get('type') == 'heading':
            last_headings.append(obj.get('value'))
            if len(last_headings) > 4:
                last_headings.pop(0)  
        if obj.get('type') == 'text':
            text_value = obj.get('md')
            page_num = obj.get('page_num')
            result.append({
                'text_value': text_value,
                'previous_headings': last_headings.copy(),  
                'text_page': page_num
            })
    
    return result

def assign_ids_to_duplicate_headings(json_list):
    heading_count = defaultdict(int)  
    updated_json_list = []
    for item in json_list:
        if item.get('type') == 'heading': 
            heading = item['value']
            heading_count[heading] += 1
            if heading_count[heading] > 1:
                item['value'] = f"{heading} |ID:{heading_count[heading]}"
        updated_json_list.append(item)
    return updated_json_list

def add_path(result_list,path_dict) :
    for ele in result_list:
        val  = ele['previous_headings'][::-1]
        print("-----ele values-------")
        print(val)
        for v in val:
            print(v)
            try:
                ele['path'] = path_dict[v]
                break
            except Exception as e:
                print(e)
    return result_list

def get_summary(text):
    sys_prompt = """You are a smart Assistant who can obtain the outputs by following the given intructions"""
    summary_prompt = f"""
    You are given the text, provide a brief summary of the Input text provided. Give the output in json format with its key value as "doc_summary"
    Here is the input text:
    {text}
    """
    summary_out = get_llm_JSONresponse(summary_prompt,sys_prompt)
    summary_out = json.loads(summary_out)
    return summary_out

load_dotenv('keys.env')
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = OPENAI_API_KEY)
parser = LlamaParse(verbose=True,api_key=LLAMA_CLOUD_API_KEY,num_workers=4)
all_files = glob.glob("./pdfs/*.pdf")
for filename in all_files:
    json_objs = parser.get_json_result(filename)
    # print(json_objs[0])
    json_list = json_objs[0]["pages"]
    md_list = []
    for j in json_list:
        md_list.append(j["md"])
    summary_json = get_summary("\n===============page===========================\n".join(md_list))
    summary_json['doc_name'] = filename
    summary_json['doc_text'] = "\n\n".join(md_list)
    heading_structure = get_document_structure(json_list)
    path_dict = {}
    key_paths = find_all_key_paths(heading_structure)
    for path in key_paths:
        val = " | ".join(path)
        path_dict[path[-1]] = val
    heading_newlist = list(path_dict.keys())
    print("new heading list is : ", heading_newlist)
    new_list = []
    page = 0
    for i in json_list:
        page+=1
        for val in i['items']:
            val['page_num'] = page
        new_list.extend(i['items'])
    print(new_list)
    new_list =assign_ids_to_duplicate_headings(new_list)
    text_heading_dict_list = extract_text_and_heading(new_list)
    print(text_heading_dict_list)
    text_heading_dict_list = add_path(text_heading_dict_list,path_dict)
    final_json = {}
    final_json['document'] = text_heading_dict_list
    final_json['document_name'] = filename.split("/")[-1].split(".pdf")[0]
    outfilename = "./json_files/"+final_json['document_name']+".json"
    summaryfilename = "./summary_json/"+final_json['document_name']+".json"

    with open(outfilename, 'w') as json_file:
        json.dump(final_json, json_file, indent=4)
    
    with open(summaryfilename, 'w') as sjson_file:
        json.dump(summary_json, sjson_file, indent=4)

