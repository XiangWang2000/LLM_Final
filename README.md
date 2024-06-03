# LLM_Final
### 下載Llama3-8B-Instruct 模型
到Hugging Face官網上先註冊帳號後，申請Llama 3模型的下載使用權，即可下載[Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  
下載完成後將其放到Meta-Llama3-8B-Instruct的資料夾中
  
### 量化模型
使用[llama.cpp](https://github.com/ggerganov/llama.cpp)來進行量化  
#### Step1 下載llama.cpp檔案並移動到該資料夾底下
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```
#### Step2 安裝所需套件
```
pip install -r requirements.txt
```
#### Step3 將前面已下載完成的Llama 3模型轉成.gguf檔
```
python3 convert-hf-to-gguf.py ../Meta-Llama3-8B-Instruct --outfile models/ggml-meta-llama-3-8b-16f.gguf
```
#### Step4 進行量化(使用Q4_K_M方法)
```
./quantize ./models/ggml-meta-llama-3-8b-16f.gguf ./models/ggml-meta-llama-3-8b-Q4_K_M.gguf Q4_K_M
```
完成模型的量化後，就要使用這個模型，來進行後續RAG的使用  

在進行RAG的使用之前，一樣需要安裝一些必要的套件
```
langchain
PyMuPDF
chromadb
sentence-transformers
llama-cpp-python
```
安裝好後就可以開始囉  
#### Step1 使用文件載入器載入要做為資料庫的文件，並使用 Text splitter 分割文件
```
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyMuPDFLoader("2.pdf")
PDF_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
```
#### Step2 載入文字Embedding model，並將embedding的結果匯入VectorDB
```
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)
```
#### Step3 在完成第一份資料的匯入後，將剩餘的資料一一追加
```
for i in range(3,10):
    loader1 = PyMuPDFLoader("{}.pdf".format(i))
    PDF_data1 = loader1.load()
    all_splits1 = text_splitter.split_documents(PDF_data1)
    vectordb.add_documents(documents=all_splits1, embedding=embedding, persist_directory=persist_directory)
```
#### Step4 設定問題
```
questions=['Give an example where F1-score is more appropriate than accuracy (actual numbers must be included, and F1-score and accuracy must be calculated separately).',
         'Explain the selection method of K in the K means algorithm.',
         'What is overfit? How to judge the occurrence of overfit? What are some possible ways to improve overfit (please write three)?',
         'What is the role of the pooling layer in CNN? Please explain the difference between max pooling and average pooling, and describe their application scenarios in CNN.',
         'Explain how one-hot encoding is done, and explain the reasons (advantages) why one-hot encoding should be used.',
         'Please give three commonly used activation functions and explain their definitions and applicable times.']
```
#### Step5 使用 LangChain 的 LlamaCpp，並增加最大輸出tokens，避免出現輸出到一半斷掉的問題
```
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="../llama.cpp/models/ggml-meta-llama-3-8b-Q4_K_M.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    max_tokens=2048,
    f16_kv=True,a
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
```
#### Step6 先測試使用Prompt的方式來測試是否能夠達到需要的效果
```
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
    You are a helpful assistant eager to assist with providing better Macchine Learning knowledge.
    <</SYS>> 
    
    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question} 
    [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Macchine Learning knowledge. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

```
#### Step7 依序詢問問題
```
prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
llm_chain = LLMChain(prompt=prompt, llm=llm)
for question in questions:
    print(question)
    llm_chain.invoke({"question": question})
```

