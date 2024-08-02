import streamlit as st
from llama_parse import LlamaParse
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from code_reader import code_reader
from prompts import context, code_parser_template
from dotenv import load_dotenv
import os
import ast

# Load environment variables
load_dotenv()

# create session variables
if 'tools' not in st.session_state:
    st.session_state['tools'] = [code_reader]

if 'code_reader_enable' not in st.session_state:
    st.session_state['code_reader_enable'] = False

if 'agent' not in st.session_state:
    st.session_state['agent'] = None

# Load large objects 
@st.cache_resource(show_spinner=False)
def getParser():
    return LlamaParse(result_type="markdown", verbose=False)

@st.cache_resource(show_spinner=False)
def getEmbeddingModel():
    return resolve_embed_model(embed_model="local:BAAI/bge-m3")

@st.cache_resource(show_spinner=False)
def getOllamaModels():
    return Ollama(model="mistral", request_timeout=300.0), Ollama(model="codellama", request_timeout=300.0)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

@st.cache_resource(show_spinner=False)
def getOutputPipeline():
    output_parser = PydanticOutputParser(CodeOutput)
    json_prompt_str = output_parser.format(code_parser_template)
    json_prompt_tmpl = PromptTemplate(template=json_prompt_str)
    return QueryPipeline(chain=[json_prompt_tmpl, llm])
 
os.makedirs("data", exist_ok=True)
parser = getParser()
embed_model = getEmbeddingModel()
llm, code_llm = getOllamaModels()
output_pipeline = getOutputPipeline()

# Create tools widget
def addTool(filenames, name, description):
    documents = parser.load_data(filenames)
    vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)
    query_engine = vector_index.as_query_engine(llm=llm)
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=name,
            description=description
        )
    )
    st.session_state['tools'].append(tool)

with st.expander(label="Create tool"):
    name = st.text_input(label="Name")
    description = st.text_input(label="Description")
    files = st.file_uploader(label="Browse files", 
                             type=['pdf', 'doc', 'docx', 'csv', 'xls', 'xlsx', 'ppt', 'pptx', 'jpg', 'jpeg', 'png'], 
                             accept_multiple_files=True,
                             help="Upload files that contain the information that is necessary to create your tool. You need at least one file to create tool.")
    if files != []:
        filenames = []
        for file in files:
            filename = os.path.join("data", file.name)
            filenames.append(filename)
            with open(filename, "wb") as f:
                f.write(file.getvalue())
        
        st.button(label="Add tool", on_click=addTool, args=(filenames, name, description, ))
    
# Create agent widget
def formatTools(tool):
    return "Name: " + tool.metadata.name + " - Description: " + tool.metadata.description

def createAgent(tools):
    if "code_reader" in [tool.metadata.name for tool in tools]:
        st.session_state['code_reader_enable'] = True
    else:
        st.session_state['code_reader_enable'] = False
    st.session_state['agent'] = ReActAgent.from_tools(tools=tools, llm=code_llm, context=context)

with st.expander(label='Create agent'):
    tools = st.multiselect(label="Choose tools",
                           options=st.session_state['tools'],
                           default=[code_reader],
                           format_func=formatTools,
                           help="Choose tools to create your agent, code_reader tool is selected by default. You need at least one tool to create agent.")

    if tools != []:
        st.button(label="Create agent", on_click=createAgent, args=(tools, ))

#Inference widget
def query(prompt):
    retries = 0
    while retries < 3:
        try:
            result = st.session_state['agent'].query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant: ", ""))
            break
        except Exception as e:
            retries += 1
            st.write(f"Error occured, retry #{retries}:", str(e))

    if retries == 3:
        st.write("Unable to process request, try again...")
    else:
        st.download_button(label="Download code file", data=cleaned_json['code'], file_name=cleaned_json['filename'])

if st.session_state['agent'] is not None:
    with st.container(border=True):
        prompt = st.text_input(label="Prompt")
        if st.session_state['code_reader_enable']:
            file = st.file_uploader(label="Upload code files", 
                                    type=['.py', '.c', '.cpp', '.js', '.cs', '.java', '.rs'],
                                    accept_multiple_files=True,
                                    help="Upload code files that you need your agent to read to generate code. You should specify reading code files in your prompt")
            if files is not None:
                for file in files:
                    filename = os.path.join("data", file.name)
                    with open(filename, "wb") as f:
                        f.write(file.getvalue())
        st.button(label="Generate code", on_click=query, args=(prompt, ))