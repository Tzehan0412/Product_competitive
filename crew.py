import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class CompetitiveAnalysisCrew:
    """產品競品分析團隊"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.gemini = LLM(
            model="gemini/gemini-3-flash-preview",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
        )
        self.search_tool = SerperDevTool()

    @agent
    def product_profiler(self) -> Agent:
        return Agent(
            config=self.agents_config["product_profiler"],
            llm=self.gemini,
            verbose=True,
        )

    @agent
    def competitor_scout(self) -> Agent:
        return Agent(
            config=self.agents_config["competitor_scout"],
            llm=self.gemini,
            tools=[self.search_tool],
            verbose=True,
        )

    @agent
    def competitor_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["competitor_analyst"],
            llm=self.gemini,
            tools=[self.search_tool],
            verbose=True,
        )

    @agent
    def strategy_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_advisor"],
            llm=self.gemini,
            verbose=True,
        )

    @task
    def product_profiling_task(self) -> Task:
        return Task(config=self.tasks_config["product_profiling_task"])

    @task
    def competitor_scouting_task(self) -> Task:
        return Task(
            config=self.tasks_config["competitor_scouting_task"],
            context=[self.product_profiling_task()],
        )

    @task
    def competitor_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["competitor_analysis_task"],
            context=[
                self.product_profiling_task(),
                self.competitor_scouting_task(),
            ],
        )

    @task
    def strategy_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_summary_task"],
            context=[
                self.product_profiling_task(),
                self.competitor_scouting_task(),
                self.competitor_analysis_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
