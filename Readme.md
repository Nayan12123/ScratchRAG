
# Policy QA (RAG Pipelines)

## Steps to Set Up the Application

### Configure `keys.env` File

1. **OpenAI API Key**: Obtain the OpenAI API key and initialise it with the `OPENAI_API_KEY` variable.
2. **Llama Cloud API Key**: Obtain the Llama Cloud API key from [Llama Index](https://cloud.llamaindex.ai/api-key). After logging in, click on "Generate New Key" and initialise it with the `LLAMA_CLOUD_API_KEY` variable. This key is used for parsing documents.
3. **Elastic Cloud API Key**: Obtain the Elastic Cloud API key from [Elastic Cloud](https://cloud.elastic.co/registration?plcmt=nav&pg=nav&tech=rpt&cta=eswt-b). First, log in and initialise it with the `ES_API_KEY` variable.
4. **Elastic Cloud ID**: Obtain the cloud ID from [Elastic Deployments](https://cloud.elastic.co/deployments/e5121bb20f2648b8a95d514ad5ec870a), and initialise it with the `ES_CLOUD_ID` variable.

---

## Use these Steps only if new files are there or the results needs to be reproduced. 
If results need not be reproduced then directly run the docker file

### Create a Conda Environment

Run the following commands in the terminal:

```bash
conda create -n pdf_qa python=3.10
conda activate pdf_qa
pip install -r requirements.txt

```

### Steps to parse documents and create index.
**step 1.** Parse the documents store in pdfs folder
```bash
python parse_files.py
```
**step 2.**  create Index. Use create_index.py to create index in the elasticvector store.
```bash
python create_index.py
```
This will create 3 indices by the following names
text_doc_index, text_summary_index, text_doc_without_headings_index


### Steps for RAG evaluation
#### create golden samples for RAG evaluation and save it.
This will create synthetic samples for RAG evaluation, further all the methods/ RAG approches' results will be computed on the golden samples and saved.
```bash
python create_eval_dataset.py
```

### Run FastAPI
#### either you can run fast api directly from the terminal or you can first deploy the model in a container and then run it
```bash
uvicorn api:app --reload
```
You can then open the browser link displayed and add /docs in the link


## Build Docker Image and run the container
If docker is not installed, install docker in the system using the [link](https://docs.docker.com/engine/install/)
```bash
cd pdf_qa
docker build -t policy_qa_image . # build docker image
docker run -d -p 8000:8000 --name qa_container policy_qa_image # build docker container
```
If a same container is running in your system first stop the container and remove it
```bash
docker stop qa_container
docker remove qa_container
```
To see the use the fast API copy and paste the following link on your browser. 
http://localhost:8000/docs#/

## File Structure
The following is an overview of the project's file structure. Each directory and file is explained to clarify its role in the project.
```
pdf_qa/
│
├── eval_data/         # Golden samples created by the LLM are saved here.
│
├── eval_methods/      # Results for the golden samples for each method are stored here.
│
├── json_files/        # Output obtained from LlamaParse, including the document structure in JSON format.
│
├── pdfs/              # PDF files passed into LlamaParse for chunking.
│
├── summary_json/      # Summaries for each document in JSON format.
│
├── api.py             # FastAPI code for each method and obtaining evaluation metrics.
│
├── create_eval_dataset.py  # Code for evaluation data preparation, golden samples, and method results.
│
├── create_index.py    # Code for indexing the documents into the Elastic Store.
│
├── default_app.py     # RAG using LlamaIndex with hybrid retrieval and default embeddings.
│
├── Dockerfile         # Dockerfile to containerise the app.
│
├── eval.py            # Code for generating evaluation metrics on the golden samples.
│
├── keys.env           # Stores all API keys.
│
├── parse_files.py     # Code to parse PDF files and structure them into JSON.
│
├── rag_main.py        # Code defining custom RAG pipelines to handle various types of queries.
│
├── pdf_qa_exp.ipynb   # Jupyter notebook for experimentation.
│
├── README.md          # Project documentation.
│
├── req_install.txt    # Pip install script for libraries.
│
├── requirements.txt   # Python dependencies for the project.
│
└── utils.py           # Custom ElasticSearch class and methods for query, search, and vector index creation.
```# ScratchRAG
