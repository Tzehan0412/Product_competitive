import os
from textwrap import dedent
from typing import Any
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from crewai_tools import SerperDevTool
from google import genai
from PIL import Image

# ============================================================
# 自定義影像生成與存檔工具
# ============================================================
class ImageGenerationTool:
    @tool("generate_and_save_commercial_image")
    def generate_and_save_commercial_image(
        prompt: str, 
        source_image_path: str, 
        output_directory: str, 
        product_name: str
    ) -> str:
        """
        呼叫 Gemini 3 Flash Image (Nano Banana 2) 進行影像生成並將結果存檔。
        
        Args:
            prompt: 繪圖提示詞，包含保護主體的權重指令。
            source_image_path: 原始圖片的檔案路徑。
            output_directory: 圖片存檔的目標資料夾路徑。
            product_name: 產品名稱，用於命名檔案。
        """
        # 確保輸出目錄存在
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        try:
            # 讀取原始圖片
            raw_image = Image.open(source_image_path)
            
            # 執行生成 (使用最新預覽模型)
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=[prompt, raw_image],
            )

            # 檔名處理：去除空格並加上後綴
            safe_name = product_name.replace(" ", "_").replace("/", "_")
            file_name = f"product_visual_{safe_name}.jpg"
            full_save_path = os.path.join(output_directory, file_name)
            
            # 解析並存檔
            for part in response.parts:
                if part.inline_data is not None:
                    generated_img = part.as_image()
                    generated_img.save(full_save_path)
                    return f"✅ 生成成功！檔案已儲存至：{full_save_path}"
            
            return "❌ 模型回應成功但未包含影像數據。"

        except Exception as e:
            return f"❌ 影像生成或存檔過程中發生錯誤: {str(e)}"

# ============================================================
# ProductVisualCrew 類別定義
# ============================================================
@CrewBase
class ProductVisualCrew:
    """通用產品視覺整合團隊"""

    agents_config = "config/agents_image.yaml"
    tasks_config = "config/tasks_image.yaml"

    def __init__(self):
        # 設定邏輯推論 LLM
        self.reasoning_llm = LLM(
            model="gemini/gemini-3-flash-preview", # 使用最新的 flash 模型進行推論
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )
        # 視覺搜尋工具
        self.search_tool = SerperDevTool()
        # 影像生成實體工具
        self.image_tool = ImageGenerationTool.generate_and_save_commercial_image

    @agent
    def visual_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config["visual_strategist"],
            llm=self.reasoning_llm,
            tools=[self.search_tool],
            verbose=True,
        )

    @agent
    def visual_prompt_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["visual_prompt_engineer"],
            llm=self.reasoning_llm,
            verbose=True,
        )

    @agent
    def image_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["image_generator"],
            llm=self.reasoning_llm,
            tools=[self.image_tool], # 核心修改：掛載存檔工具
            verbose=True,
        )

    @task
    def visual_strategy_task(self) -> Task:
        return Task(config=self.tasks_config["visual_strategy_task"])

    @task
    def visual_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config["visual_prompt_task"],
            context=[self.visual_strategy_task()],
        )

    @task
    def image_generation_task(self) -> Task:
        """
        此任務現在會驅動 Agent 使用 generate_and_save_commercial_image 工具。
        """
        return Task(
            config=self.tasks_config["image_generation_task"],
            context=[self.visual_prompt_task()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )