from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
import glob
import os
import json
from dotenv import load_dotenv
from rag_main import *

load_dotenv("keys.env")
os.environ["OPENAI_API_KEY"]  = os.getenv('OPENAI_API_KEY')
pdf_files = glob.glob("./pdfs/*.pdf")
dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(
    document_paths=pdf_files,
    max_goldens_per_document=4,
    synthesizer = Synthesizer(model="gpt-4o"),
    include_expected_output = True
)
saved_path = dataset.save_as(
    file_type='json',  # or 'csv'
    directory="./eval_data", 
)


get_results_syntheticData(method_1_with_summary)
get_results_syntheticData(method_2_with_reranking)
get_results_syntheticData(method_3_with_RRF)
get_results_syntheticData(method_4_without_heading_index)
get_results_syntheticData(method_5_without_doc_filter)

