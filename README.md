# LLM_Final
## 下載Llama3-8B-Instruct 模型
到Hugging Face官網上先註冊帳號後，申請Llama 3模型的下載使用權，即可下載[Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  
下載完成後將其放到Meta-Llama3-8B-Instruct的資料夾中
  
## 量化模型
使用[llama.cpp](https://github.com/ggerganov/llama.cpp)來進行量化  
* Step 1 下載llama.cpp檔案並移動到該資料夾底下
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```
* Step 2 安裝所需套件
```
pip install -r requirements.txt
```
* Step 3 將前面已下載完成的Llama 3模型轉成.gguf檔
```
python3 convert-hf-to-gguf.py ../Meta-Llama3-8B-Instruct --outfile models/ggml-meta-llama-3-8b-16f.gguf
```
* Step 4 進行量化(使用Q4_K_M方法)
```
./quantize ./models/ggml-meta-llama-3-8b-16f.gguf ./models/ggml-meta-llama-3-8b-Q4_K_M.gguf Q4_K_M
```
完成模型的量化後，就要使用這個模型，來進行後續RAG的使用  
若你覺得前面的操作太過麻煩的話，也可以直接下載我已經[量化好的模型](https://huggingface.co/xiangw21/ggml-meta-llama-3-8b-Q4_K_M.gguf)進行使用

## RAG with Llama 3
在進行RAG的使用之前，一樣需要安裝一些必要的套件
```
langchain
PyMuPDF
chromadb
sentence-transformers
llama-cpp-python
```
安裝好後就可以開始囉  
* Step 1 使用文件載入器載入要做為資料庫的文件，並使用 Text splitter 分割文件
```
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyMuPDFLoader("2.pdf")
PDF_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
```
* Step 2 載入文字Embedding model，並將embedding的結果匯入VectorDB
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
* Step 3 在完成第一份資料的匯入後，將剩餘的資料一一追加
```
for i in range(3,10):
    loader1 = PyMuPDFLoader("{}.pdf".format(i))
    PDF_data1 = loader1.load()
    all_splits1 = text_splitter.split_documents(PDF_data1)
    vectordb.add_documents(documents=all_splits1, embedding=embedding, persist_directory=persist_directory)
```
* Step 4 設定問題
```
questions=['Give an example where F1-score is more appropriate than accuracy (actual numbers must be included, and F1-score and accuracy must be calculated separately).',
         'Explain the selection method of K in the K means algorithm.',
         'What is overfit? How to judge the occurrence of overfit? What are some possible ways to improve overfit (please write three)?',
         'What is the role of the pooling layer in CNN? Please explain the difference between max pooling and average pooling, and describe their application scenarios in CNN.',
         'Explain how one-hot encoding is done, and explain the reasons (advantages) why one-hot encoding should be used.',
         'Please give three commonly used activation functions and explain their definitions and applicable times.']
```
* Step 5 使用 LangChain 的 LlamaCpp，並增加最大輸出tokens，避免出現輸出到一半斷掉的問題
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
* Step 6 先測試使用Prompt的方式來測試是否能夠達到需要的效果
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
* Step 7 依序詢問問題
```
prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
llm_chain = LLMChain(prompt=prompt, llm=llm)
for question in questions:
    print(question)
    llm_chain.invoke({"question": question})
```
* Step 8 接著使用剛剛建立好的資料庫當作Retriever
```
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```
* Step 9 依序詢問問題，以測試RAG的效果與單純使用Prompt的效果差異
```
for question in questions:
    print(question)
    qa.invoke(question)
```

## Example 1
#### Question: Give an example where F1-score is more appropriate than accuracy (actual numbers must be included, and F1-score and accuracy must be calculated separately).
#### Answer(Prompt): 
>    To illustrate a scenario where F1-score is more suitable than accuracy, let's consider the following example:
>
>  A bank has implemented a machine learning model to predict whether an applicant will repay their loan on time. The model has been trained on historical data and has produced the following results:
>
>  | Predicted Outcome | Actual Outcome |
>  | --- | --- |
>  | 0 (default) | 1 (repaid) |
>  | 0 (default) | 0 (default) |
>  | 1 (repaid) | 1 (repaid) |
>
>  In this scenario, the model has produced an accuracy of 80% (8 correct predictions out of 10). However, when we look at the precision and recall values, we can see that they are quite different.
>
>  The precision for predicting a positive outcome is 90% (9 true positives out of 10). This means that out of all the positive predictions made by the model, 90% of them were actually correct.
>
>  On the other hand, the recall for predicting a positive outcome is only 60% (6 actual positive outcomes out of 10). This means that out of all the actual positive outcomes in the data, 40% of them were not correctly predicted by the model.
>
>  In this scenario, the accuracy is higher than the F1-score. The accuracy measures how well the model can predict both positive and negative outcomes. However, since the model is biased towards predicting more positive outcomes (as evident from the high precision value), it tends to overlook some of the actual positive outcomes in the data.
>
>  In this case, a lower F1-score indicates that the model's predictions are less accurate overall. This can be attributed to the model's bias towards predicting more positive outcomes, which leads to a higher number of false positives and missed actual positive outcomes.
>
>  Therefore, in this scenario, an F1-score would be a more appropriate measure of accuracy than the simple accuracy metric used earlier. The F1-score takes into account both precision and recall, providing a more comprehensive picture of the model's performance.
#### Answer(RAG): 
> Example:
A hospital wants to evaluate the effectiveness of its new diagnosis system. The system correctly identifies 90% of positive cases (True Positives = 0.9). At the same time, it incorrectly rejects 10% of negative cases (False Negatives = 0.1).
>
>In this scenario, accuracy would be:
Accuracy = (TP + TN) / (TP + FN + TN + FN)
= (90 + 80) / (90 + 10 + 80 + 20)
= 170/280
= 0.607
>
>F1-score would be:
F1 = 2 * ((TP)/(TP+FN)) * ((TN)/(TN+FP)))
= 2 * ((0.9)/(0.9+0.1))) * ((0.8)/(0.8+0.2)))
= 2 * (0.9/1) * (0.8/0.4))
= 2 * 0.9 * 2
= 3.6
>
>In this example, F1-score is more appropriate than accuracy because it takes into account both precision and recall. The F1-score of 0.36 indicates that the diagnosis system is not very effective in identifying positive cases (it has a low precision) or in avoiding false positives (it has a high recall). Therefore, the F1-score provides a more comprehensive evaluation of the performance of the diagnosis system than accuracy alone. 
>
>If you don't know the answer, just say that you don't know and don't try to make up an answer.
>
>In this case, I will say that I don't know if there is another way to evaluate the performance of a diagnostic system. If someone knows, please share!
## Example 2
#### Question: Explain the selection method of K in the K means algorithm.
#### Answer(Prompt): 
>  I'd be happy to explain the selection method of K in the K-means algorithm.
>  The selection method for K in the K-means algorithm is often referred to as "random initialization" or "k-random initialization". This approach involves selecting k initial centroids at random from the dataset.
>
>This method has been widely used and studied in the context of k-means clustering. However, it's worth noting that other methods have also been proposed for selecting k in the K-means algorithm.
>
>I hope this explanation helps you understand the selection method for k in the K-means algorithm. Let me know if you have any further questions!
#### Answer(RAG): 
>In the K-Means clustering algorithm, the value of K (the number of clusters) is typically determined using a combination of theoretical considerations and empirical experimentation.
>
>In general, there are no hard-and-fast rules for selecting the optimal value of K. Instead, practitioners often rely on various heuristics and guidelines to help guide their selection.
>
>Some common approaches for selecting K include:
>
>1. Visual inspection: Examining the clustering results visually can provide insight into whether the chosen value of K is appropriate.
>2. Silhouette analysis: This technique involves calculating a score for each sample in the dataset, based on its proximity to the nearest cluster center. Samples with high silhouette scores tend to be well-clustered, while those with low scores may be misclustered.
>3. Gap statistics: This approach involves computing the average distance between samples within each cluster, as well as the minimum distance between any two clusters. The gap statistic is then computed as the >difference between these two distances. A higher gap statistic indicates that the chosen value of K is a good fit for the data.
>
>By combining these approaches with domain-specific knowledge and expertise, practitioners can develop effective strategies for selecting the optimal value of K in their specific applications.
## Example 3
#### Question: What is overfit? How to judge the occurrence of overfit? What are some possible ways to improve overfit (please write three)?
#### Answer(Prompt):
>    Overfit occurs when a machine learning model performs exceptionally well on the training data but fails to generalize well to new, unseen data. This is often due to the model being too complex and memorizing the training data rather than learning meaningful patterns.
>
>To judge the occurrence of overfit, you can use techniques such as:
>
>1. Cross-validation: By splitting your data into multiple subsets, you can train your model on each subset and evaluate its performance.
>2. Model complexity reduction: You can try to reduce the complexity of your model by removing unnecessary features or using regularization techniques.
>
>Some possible ways to improve overfit are:
>
>1. Regularization: This involves adding a penalty term to the loss function that discourages complex models from being trained.
>2. Early stopping: This involves monitoring the performance of the model on a validation set and stopping training early if the model's performance starts to degrade.
>3. Ensemble methods: This involves combining the predictions of multiple models, either by taking an average or by using a more sophisticated method such as bagging or boosting. By combining the strengths of multiple models, ensemble methods can often improve the overall performance of the system.
#### Answer(RAG):
>Overfitting is a situation in which a machine learning model becomes too complex for its own good. This often happens when you have a lot of data and your model tries to fit every detail.
>
>To judge whether your model has become overfitted, you can check the performance of your model on a test set that it hasn't seen before. If your model performs significantly better on the training set than on the test set, then it's likely that your model is overfitted.
>
>Here are three possible ways to improve overfit:
>
>1. **Regularization**: You can add a regularization term to your loss function to discourage large weights and prevent overfitting.
>2. **Data augmentation**: If you have limited data, you can try generating additional training examples through techniques such as rotation, flipping, or color jittering. This can help your model generalize better to new data and reduce the risk of overfitting.
>3. **Early stopping**: You can also try early stopping by monitoring the performance of your model on a validation set during training. If you notice that the performance is no longer improving (or even degrading) after some point, then it's likely that your model has already reached its optimal level of accuracy and further training would only lead to overfitting.

## 結論
由以上範例可知，有些題目經過RAG之後，可以得到比較好和比較精確的答案，但如果是比較常是一點的問題的話，不需使用RAG也能夠得到很好的效果。
因此，這個`Llama 3 + RAG`的組合非常適合應用在期中末考，老師有開放查看簡報但是斷網的上機考
