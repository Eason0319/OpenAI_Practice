import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# LangChain 相關
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
# 安全性 (Swagger UI 鎖頭)
from fastapi.security import APIKeyHeader

# 讀取 .env
load_dotenv()

# --- 全域設定 ---
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ntuedtd")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "secret")

# --- 初始化 Pinecone ---
# 這裡我們只建立一個 vectorstore 物件，不用每次請求都重連
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# 定義 RAG 的 Prompt 模板
# {context} 會被替換成從 Pinecone 撈出來的資料
RAG_SYSTEM_PROMPT = """你是一個非常了解富邦悍將棒球隊資訊的分析師，請只回答參考資料中有的答案，如果有不知道答案的問題請誠實回答"我只知道富邦悍將的相關訊息"。
【參考資訊】：
{context}
"""

# --- FastAPI 設定 ---
app = FastAPI(title="RAG Chatbot with Pinecone")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全性設定
header_scheme = APIKeyHeader(name="x-secret-token", auto_error=False)

# --- 資料模型 ---
class ChatRequest(BaseModel):
    user: str
    model: str = LLM_MODEL

class LoginRequest(BaseModel):
    password: str

# --- 驗證函式 ---
async def verify_token(api_key: str = Depends(header_scheme)):
    if api_key != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid Token")
    return api_key

# --- 路由 ---

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post("/login")
def login(req: LoginRequest):
    if req.password == ADMIN_PASSWORD:
        return {"token": ADMIN_PASSWORD, "status": "success"}
    else:
        raise HTTPException(status_code=401, detail="密碼錯誤")

@app.post("/chat", dependencies=[Depends(verify_token)])
def chat(req: ChatRequest):
    # 1. 定義 Retriever (檢索器)
    # search_kwargs={"k": 3} 代表只找最相關的 3 筆資料
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 定義 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", "{question}")
    ])

    # 3. 定義 LLM
    llm = ChatOpenAI(model=req.model, temperature=0.3)

    # 4. 定義 RAG Chain (鍊)
    # 這裡的邏輯是：
    # 步驟 A: 把 "question" 丟給 retriever 找資料，填入 "context"
    # 步驟 B: 把 "question" 直接傳遞下去
    # 步驟 C: 組合 Prompt -> 丟給 LLM -> 解析字串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 執行
    # invoke 的輸入會同時被傳給 retriever (去查 context) 和 prompt (填入 question)
    result = rag_chain.invoke(req.user)
    
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)