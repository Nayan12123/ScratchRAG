from utils import*
index_list = CustomES().get_all_index_names()      
print("all_indices are: ", index_list)
text_es = CustomES(index="text_doc_index")
# # text_es.delete_index("text_doc_index")
# # text_es.delete_index("text_summary_index")

text_list,page_list,doc_namelist = get_textNodeList_with_headers()

print("type of text:", type(text_list[0]))
print("type of page:", type(page_list[0]))
print("type of doc_name:", type(doc_namelist[0]))




text_es.create_text_index(index_name="text_doc_index",text_list=text_list,page_list=page_list,doc_name_list=doc_namelist)

text_list,summmary_list,doc_namelist= get_summaryNodeList()
summary_es = CustomES()
summary_es.create_summary_index(index_name="text_summary_index",doc_text_list=text_list,summary_list=summmary_list,doc_name_list=doc_namelist)

text_es = CustomES(index="text_doc_without_headings_index")
text_list,page_list,doc_namelist = get_textNodeList()
text_es.create_text_index(index_name="text_doc_without_headings_index",text_list=text_list,page_list=page_list,doc_name_list=doc_namelist)

