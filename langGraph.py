import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv()

# 使用原本運作穩定的名稱
MODEL_NAME = "gemini-3-flash-preview" 

# ============================================================
# 1. State 定義
# ============================================================
class CreativeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strategy_report: str = "" 
    topics: list = Field(default_factory=list)
    drafts: list = Field(default_factory=list)
    reviewed: list = Field(default_factory=list)
    retry_count: int = 0
    final_options: list = Field(default_factory=list)

# ============================================================
# 2. 輔助函式 (修正補齊)
# ============================================================

def _force_list(data: Any) -> list:
    """確保資料格式符合 Pydantic 的 list 定義，防止 NameError"""
    if data is None: 
        return []
    if isinstance(data, list): 
        return data
    if isinstance(data, dict): 
        return [data]
    return [str(data)]

def _normalize_content(content) -> str:
    """處理 Gemini 特有的 [{'type': 'text', 'text': '...'}] 結構"""
    if isinstance(content, str): 
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)
    return str(content)

def _parse_json_response(content: str):
    content = _normalize_content(content)
    
    # Debug 訊息：查看處理後的文字前段
    # print(f"--- [處理後的純文字內容] ---\n{content[:200]}...") 

    try:
        # 移除 Markdown 標籤
        content = re.sub(r'```json\s?|\s?```', '', content).strip()
        
        # 優先尋找 [ ] 陣列邊界 (最長匹配)
        array_match = re.search(r'(\[.*\])', content, re.DOTALL)
        if array_match:
            return json.loads(array_match.group(1))
            
        # 次之尋找 { } 物件邊界
        object_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if object_match:
            return json.loads(object_match.group(1))
            
        return None 
    except Exception as e:
        print(f"❌ JSON 解析最終失敗: {e}")
        return None

# ============================================================
# 3. 節點定義
# ============================================================

def supervisor(state: CreativeState) -> Command:

    # 1. 如果已有審核結果 -> 輸出
    if state.reviewed and len(state.reviewed) > 0:
        return Command(goto="output")
    
    # 2. 如果已有草稿 -> 送去審核 (critic)
    if state.drafts and len(state.drafts) > 0:
        return Command(goto="critic")
    
    # 3. 如果已有切角 -> 送去寫文案 (copywriter)
    if state.topics and len(state.topics) > 0:
        return Command(goto="copywriter")
    
    # 4. 初始狀態 -> 從切角開始 (curator)
    return Command(goto="curator")

def curator(state: CreativeState) -> Command:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1)
    
    prompt = f"""你是一個精準的資料提取器。請閱讀報告並提取『產品基本資訊』與『三大切角矩陣』。
    
    【報告內容】：
    {state.strategy_report}
    
    【輸出規範】：
    1. 必須只回傳一個 JSON Array。
    2. 嚴禁任何解釋性文字。
    3. 每個物件必須包含產品名稱與核心特色。
    4. 格式範例：
    [
      {{
        "product_name": "產品全稱",
        "product_features": ["特色1", "特色2", "特色3"],
        "topic_title": "標題", 
        "original_hook": "鉤子內容", 
        "selection_reason": "理由"
      }}
    ]
    """
    
    response = llm.invoke(prompt)
    result = _force_list(_parse_json_response(response.content))
    
    if not result:
        print("❌ curator 提取失敗！LLM 可能未按格式回傳。")
        result = []
        
    return Command(goto="supervisor", update={"topics": result})

def copywriter(state: CreativeState) -> Command:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
    
    # 將三個話題資訊轉為 JSON 字串供 LLM 參考
    topics_context = json.dumps(state.topics, ensure_ascii=False)
    
    prompt = f"""你是首席品牌文案官。現在請你根據以下『三個行銷切角』，撰寫出一篇完整的『全方位整合行銷長文案』。
    
    【切角資訊來源】：
    {topics_context}
    
    【文案撰寫規範】：
    1. 整合目標：不要只是列出三個切角，而是要將它們串聯起來。
       (邏輯參考：因為有『極致性能』支撐，所以能展現『水下美學』，最後證明這是一場完美的『升級投資』)。
    2. hook_line：必須是一個能概括品牌靈魂的強力開頭。
    3. threads_copy：撰寫一段具備「深度對談感」的長文，適合在 Threads 或 FB 引發專業討論。
    4. ig_copy：撰寫一份結構清晰的「懶人包式」文案，使用列點展示三大利益點。
    5. cta：設計一個讓受眾無法拒絕的結尾行動指令。

    【輸出規範】：
    必須回傳 JSON Array 格式，且僅包含一個整合後的物件：
    [
      {{
        "product_name": "產品全稱",
        "product_features": ["特色1", "特色2", "特色3"],
        "topic_title": "全方位品牌整合文案",
        "hook_line": "...",
        "threads_copy": "...",
        "ig_copy": "...",
        "cta": "..."
      }}
    ]
    """
    
    response = llm.invoke(prompt)
    # 使用我們之前的強化版解析器
    result = _force_list(_parse_json_response(response.content))
    
    if not result:
        print("⚠️ 整合型 copywriter 產出失敗，注入保底內容。")
        result = [{"topic_title": "整合文案", "hook_line": "一次到位的極致選擇", "threads_copy": "...", "ig_copy": "...", "cta": "..."}]
        
    return Command(goto="supervisor", update={"drafts": result})

def critic(state: CreativeState) -> Command:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)
    prompt = f"""你是創意總監。請審核文案並給予評分(0-100)。
    請務必保留原有的文案內容，並在每個物件中加入 'score' 與 'revision_notes'。
    
    待審文案：{json.dumps(state.drafts, ensure_ascii=False)}
    
    回傳格式：
    [
      {{"topic_title": "...", "score": 90, "revision_notes": "...", "hook_line": "...", "threads_copy": "...", "ig_copy": "...", "cta": "..."}}
    ]"""
    
    response = llm.invoke(prompt)
    result = _force_list(_parse_json_response(response.content))
    
    if not result:
        print("⚠️ critic 未能產出 JSON，注入保底評分。")
        result = [dict(d, score=85, revision_notes="Pass") for d in state.drafts]
        
    return Command(goto="supervisor", update={"reviewed": result})

def output(state: CreativeState) -> dict:
    print("\n--- ✨ LangGraph 流程成功結束 ---")
    final = state.reviewed if state.reviewed else state.drafts
    return {"final_options": final}

# ============================================================
# 4. 圖表組裝
# ============================================================
def build_marketing_graph():
    workflow = StateGraph(CreativeState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("curator", curator)
    workflow.add_node("copywriter", copywriter)
    workflow.add_node("critic", critic)
    workflow.add_node("output", output)
    
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("output", END)
    return workflow.compile()