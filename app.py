import os
from dotenv import load_dotenv # For managing API keys, though not strictly needed for just loading

import streamlit as st

# Import libraries:
from langchain_community.documemt_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import <embedding model >
from langchain.vectorstores import <vector model>
from langchain.llms import <llm model>
from langchain.chains import RetreivalQA

DATA_PATH  = "documents/"

def load_documents():
  
