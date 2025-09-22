import multiprocessing as mp
import os
from dotenv import load_dotenv

from agent import Agent
from config import cfg
from interpreter_module import Interpreter
from llm_client import LLMClient
from models import Journal
from utils import save_run

# 加载环境变量
load_dotenv()


def main():
    def exec_callback(*args, **kwargs):
        res = interpreter.run(*args, **kwargs)
        return res

    journal = Journal()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY环境变量")
    
    llm = LLMClient(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )
    agent = Agent(cfg=cfg, journal=journal, llm=llm)

    interpreter = Interpreter()

    global_step = len(journal)
    while global_step < cfg.agent.steps:
        agent.step(exec_callback=exec_callback)
        save_run(cfg, journal)
        global_step = len(journal)

    interpreter.cleanup_session()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
