import os
from textwrap import dedent
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from google import genai
from PIL import Image

@CrewBase
class ProductVisualCrew:
    """通用產品視覺整合團隊"""

    agents_config = "config/agents_image.yaml"
    tasks_config = "config/tasks_image.yaml"

    def __init__(self):
        self.reasoning_llm = LLM(
            model="gemini/gemini-3-flash-preview",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )
        # 視覺搜尋工具
        self.search_tool = SerperDevTool()

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
        # 此任務涉及實際的 SDK 呼叫，建議在 callback 或自定義工具中執行
        return Task(
            config=self.tasks_config["image_generation_task"],
            context=[self.visual_prompt_task()],
            # 這裡可以透過特定 function 實作原 Code 的 client.models.generate_content
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )