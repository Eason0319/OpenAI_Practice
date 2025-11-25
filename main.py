import os
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 讀取 .env 檔案
load_dotenv()

# 設定全域變數
LLM_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"
# 讀取我們剛設定的密碼
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "secret")

# 定義請求模型
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

class LoginRequest(BaseModel):
    password: str

app = FastAPI(title="LC + OpenAI: Secured Chat")

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 驗證函數 (Security Dependency)
# 每個受保護的請求都需要在 Header 帶上 x-secret-token
async def verify_token(x_secret_token: str = Header(...)):
    # 這裡做一個簡單的驗證：Token 必須等於我們的密碼
    # (在生產環境通常會發行 JWT，但個人練習這樣最快且有效)
    if x_secret_token != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid Token")
    return x_secret_token

@app.get("/")
def read_root():
    return FileResponse("index.html")

# [New] 登入 API
@app.post("/login")
def login(req: LoginRequest):
    if req.password == ADMIN_PASSWORD:
        # 登入成功，回傳密碼作為 Token (或是你可以生成隨機亂碼並存起來)
        return {"token": ADMIN_PASSWORD, "status": "success"}
    else:
        raise HTTPException(status_code=401, detail="密碼錯誤")

# [Modified] 聊天 API，加入 verify_token 依賴
@app.post("/chat")
def chat(req: ChatRequest, token: str = Depends(verify_token)):
    sys_merged = DEFAULT_SYSTEM_PROMPT if req.system == DEFAULT_SYSTEM_PROMPT \
                 else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user", "{question}")
    ])

    llm = ChatOpenAI(
        model=req.model,
        temperature=0.3
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"question": req.user})
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)