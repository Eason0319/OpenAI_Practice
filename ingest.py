import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# 讀取 .env 設定
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragpractice")
DATA_FILE = "data.txt"  # 指定你的資料檔名

def main():
    # 1. 檢查資料檔是否存在
    if not os.path.exists(DATA_FILE):
        print(f"❌ 錯誤：找不到 {DATA_FILE} 檔案，請確認檔案位置。")
        return

    print(f"📂 正在讀取 {DATA_FILE}...")

    # 2. 讀取文字檔內容
    # 這裡我們假設每一行都是一筆獨立的知識
    documents = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 忽略空行
                documents.append(Document(page_content=line))

    if not documents:
        print("⚠️ 檔案是空的，沒有資料可以上傳。")
        return

    print(f"🔄 準備上傳 {len(documents)} 筆資料到 Pinecone: {INDEX_NAME}...")

    # 3. 初始化 Embeddings 模型
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. 上傳到 Pinecone
    try:
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print("✅ 上傳成功！資料已存入向量資料庫。")
    except Exception as e:
        print(f"❌ 上傳失敗: {e}")

if __name__ == "__main__":
    main()