import os
from openagi.actions.files import WriteFileAction
from openagi.actions.tools.ddg_search import DuckDuckGoNewsSearch
from openagi.actions.tools.webloader import WebBaseContextTool
from openagi.agent import Admin
from openagi.llms.groq import GroqModel
from openagi.memory import Memory
from openagi.planner.task_decomposer import TaskPlanner
from openagi.worker import Worker
from rich.console import Console
from rich.markdown import Markdown


os.environ['GROQ_API_KEY'] = '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'
os.environ['GROQ_MODEL'] = 'llama-3.1-70b-versatile'
os.environ['GROQ_TEMP'] = '0.3'

groq_config = GroqModel.load_from_env_config()
llm = GroqModel(config=groq_config)


analysis_worker = Worker(
    role="Analyst",
    instructions="""
    1. Analyze the potential market size, demand, and profitability of the idea or product.
    2. Identify key competitors, their strengths and weaknesses, and analyze their market positioning.
    3. Evaluate potential risks and challenges associated with the idea or product.
    Gather and assess relevant data to determine the market's potential, competitive threats, and risks.
    """,
    actions=[DuckDuckGoNewsSearch, WebBaseContextTool],
    llm=llm,  
    max_iterations=1,
)


strategy_worker = Worker(
    role="Strategist",
    instructions="""
    1. Provide actionable recommendations to enhance the idea or product.
    2. Predict future outcomes based on current market data and trends.
    Use historical data, trend analysis, and strategic thinking to offer insights and forecasts.
    """,
    actions=[DuckDuckGoNewsSearch, WebBaseContextTool],
    llm=llm,  
    max_iterations=1,
)


writer_reviewer_worker = Worker(
    role="Writer and Reviewer",
    instructions="""
    1. Compose a comprehensive report that integrates findings from all analyses.
    2. Review the composed report for clarity, accuracy, and completeness.
    3. Finalize the report and save it to a file.
    """,
    actions=[WriteFileAction],
    llm=llm,  
    max_iterations=1,
    force_output=True,
)


admin = Admin(
    planner=TaskPlanner(human_intervene=False),
    memory=Memory(),
    llm=llm,  
)


admin.assign_workers([
    analysis_worker,
    strategy_worker,
    writer_reviewer_worker
])


res = admin.run(
    query="Evaluate a startup idea focused on developing an AI-powered fitness app that provides personalized workout plans and tracks progress through machine learning.",
    description="Analyze the market potential, competition, risks, provide actionable insights, and forecast future outcomes.",
)


print("-" * 100)  # Separator
Console().print(Markdown(res))
