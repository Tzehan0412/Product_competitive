import os
import io
import json
from textwrap import dedent
from typing import Optional, List
from PIL import Image

from google import genai
from google.genai import types
from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start
from crewai.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

import opik
from opik.integrations.crewai import track_crewai

# 載入環境變數
load_dotenv()
# 追蹤專案名稱
track_crewai(project_name="Universal-Product-Visual-Master")

# ============================================================
# 區塊 1：簡化後的資料結構定義 (僅保留產品名稱與文案)
# ============================================================

class MarketingContext(BaseModel):
    """資料來源嚴格限定於 main.py 傳入的資訊"""
    product_name: str = Field(..., description="產品名稱")
    copywriting: str = Field(..., description="行銷文案/Hook")
    source_image_path: str = Field(..., description="原始圖片路徑")

class ImageSearchQueries(BaseModel):
    queries: List[str]
    reasoning: str

class NanoBananaPrompt(BaseModel):
    """專為 Nano Banana 2 設計的保護型 Prompt"""
    final_prompt: str = Field(..., description="最終繪圖指令，包含對核心產品主體的保護指令")

# ============================================================
# 區塊 2：視覺搜尋工具
# ============================================================

@tool("visual_search_tool")
def visual_search_tool(query: str) -> str:
    """搜尋產品視覺背景與商業攝影氛圍素材。"""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=f"{query} professional commercial photography background aesthetic", max_results=3)
    return str(results)

# ============================================================
# 區塊 3：通用產品視覺整合 Flow (自主視覺發想架構)
# ============================================================

class UniversalProductFlow(Flow):

    @start()
    def analyze_marketing_strategy(self):
        """步驟 1: 根據產品名稱與文案，由 AI 自主決定視覺風格"""
        ctx: MarketingContext = self.state["marketing_context"]
        print(f"🔍 [Step 1] AI 正在為 {ctx.product_name} 構思最契合文案的視覺風格...")
        
        strategist = Agent(
            role="Commercial Visual Strategist",
            goal=f"根據產品 '{ctx.product_name}' 與文案 '{ctx.copywriting}' 自主發想拍攝場景",
            backstory="你是頂尖商業攝影指導，擅長將抽象文案轉化為具體的視覺構成、材質與光影佈局。",
            llm="gemini/gemini-2.5-flash"
        )

        task = Task(
            description=dedent(f"""
                分析產品：{ctx.product_name}
                文案重點：{ctx.copywriting}
                
                任務：
                1. 根據文案氛圍，自主決定最能襯托產品價值的「場景」、「背景材質」與「光影風格」。
                2. 產出 3 個搜尋關鍵字，聚焦於環境氛圍，嚴禁修改產品主體結構。
            """),
            agent=strategist,
            output_pydantic=ImageSearchQueries,
            expected_output="產品商業攝影視覺關鍵字"
        )

        result = Crew(agents=[strategist], tasks=[task]).kickoff()
        self.state["search_queries"] = result.pydantic.queries
        return self.state["search_queries"]

    @listen(analyze_marketing_strategy)
    def fetch_visual_inspiration(self):
        """步驟 2: 抓取視覺參考素材"""
        print(f"🚀 [Step 2] 正在根據 AI 構思搜尋視覺靈感...")
        materials = []
        for q in self.state["search_queries"]:
            res = visual_search_tool.run(query=q)
            materials.append(res)
        self.state["visual_references"] = "\n".join(materials)
        return self.state["visual_references"]

    @listen(fetch_visual_inspiration)
    def design_protected_prompt(self):
        """步驟 3: 撰寫「主體絕對保護」導向的 Prompt"""
        print("🎨 [Step 3] 撰寫 Nano Banana 2 產品完整性保護提示詞...")
        ctx: MarketingContext = self.state["marketing_context"]

        engineer = Agent(
            role="Subject-Integrity Prompt Engineer",
            goal="生成能完美保留原圖產品細節，僅針對背景進行創意替換的 Prompt",
            backstory="你是 AI 影像生成專家，擅長利用權重指令確保產品輪廓、文字、材質 100% 原始呈現。",
            llm="gemini/gemini-2.5-flash"
        )

        task = Task(
            description=dedent(f"""
                請為產品「{ctx.product_name}」撰寫 i2i (Image-to-Image) 繪圖指令。
                
                【核心保護指令 - 權重最高】:
                - (Original primary subject from source image:1.9): 必須完整保留原圖產品的輪廓、標籤、質地。
                - (Maintain object structural integrity:1.8): 禁止任何變形或重新繪製產品。
                
                【場景生成描述】:
                - 環境設定: 參考靈感素材 {self.state['visual_references']}
                - 氛圍對齊: 呼應文案靈魂「{ctx.copywriting}」。
                
                請輸出一個高品質的英文 Prompt。
            """),
            agent=engineer,
            output_pydantic=NanoBananaPrompt,
            expected_output="包含主體保護權重的 Nano Banana 2 Prompt"
        )

        result = Crew(agents=[engineer], tasks=[task]).kickoff()
        self.state["final_prompt"] = result.pydantic.final_prompt
        return self.state["final_prompt"]

    @listen(design_protected_prompt)
    def execute_image_generation(self):
        """步驟 4: 呼叫 Nano Banana 2 (Gemini 3 Flash Image) 執行生成"""
        print("📸 [Step 4] 執行商業級背景替換 (AI 自主風格)...")
        ctx: MarketingContext = self.state["marketing_context"]
        final_prompt = self.state["final_prompt"]

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        try:
            image = Image.open(ctx.source_image_path)
            
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[final_prompt, image],
            )

            output_filename = f"product_visual_{ctx.product_name.replace(' ', '_')}.jpg"
            
            for part in response.parts:
                if part.inline_data is not None:
                    generated_img = part.as_image()
                    generated_img.save(output_filename)
                    self.state["final_image_result"] = output_filename
                    print(f"✅ 生成成功！檔案已儲存：{output_filename}")

        except Exception as e:
            print(f"❌ 生成失敗: {e}")
            self.state["final_image_result"] = None

    @listen(execute_image_generation)
    def finalize_report(self):
        print("\n" + "🚀" * 20)
        print("Universal Product Visual Flow Completed")
        print(f"產品名稱: {self.state['marketing_context'].product_name}")
        print(f"生成的視覺檔案: {self.state.get('final_image_result')}")
        print("🚀" * 20 + "\n")

# ============================================================
# 執行區域：直接讀取 marketing_result.json
# ============================================================
if __name__ == "__main__":
    json_path = "marketing_result.json"
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 取得 langGraph 產出的第一個結果
            marketing_info = data["final_options"][0]

        # 將 JSON 內容直接映射到 MarketingContext，移除場景與天氣
        marketing_data = MarketingContext(
            product_name=marketing_info.get("product_name"),
            copywriting=marketing_info.get("hook_line"),
            source_image_path="./1233.jpg" # 請確保此處為正確的原始圖路徑
        )

        flow = UniversalProductFlow()
        flow.state["marketing_context"] = marketing_data
        flow.kickoff()
    else:
        print(f"❌ 找不到 {json_path}，請確認檔案是否存在。")