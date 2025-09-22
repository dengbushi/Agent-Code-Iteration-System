import random
from typing import Callable

from pydantic import BaseModel, Field

from interpreter_module import ExecutionResult
from llm_client import LLMClient
from models import Journal, Node
from text_utils import extract_code, extract_text_up_to_code, wrap_code, extract_jsons
from utils import data_preview_generate


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class Evaluation(BaseModel):
    is_buggy: bool = Field(...)
    metric: float | None = Field(...)
    summary: str = Field(...)


class Agent:
    def __init__(self, cfg, journal: Journal, llm: LLMClient):
        self.cfg = cfg
        self.journal = journal
        self.data_preview: str | None = None
        self.llm = llm

    def search_policy(self) -> Node | None:
        search_cfg = self.cfg.agent.search

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [n for n in self.journal.buggy_nodes if n.is_leaf]
            if debuggable_nodes:
                return random.choice(debuggable_nodes)

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            return None
        return self.journal.get_best_node()

    def plan_and_code_query(self, system_message, user_message, retries=3) -> tuple[str, str]:
        completion_text = None
        for _ in range(retries):
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
            )
            print("--- Start of LLM Response ---")
            print(response)
            print("--- End of LLM Response ---")
            completion_text = response
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)
            if code:
                return nl_text, code
            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text

    def _draft(self) -> Node:
        system_prompt = "You are an AI agent."
        user_prompt = [
            "You have to come up with a solution for machine learning task and then implement this solution in Python.",
            f"The task is to {str(self.cfg.task_goal)} ",
            f'All the provided input data is stored in "{self.cfg.data_dir}" directory.',
            '**IMPORTANT**: When loading data, you MUST use the provided path variable `data_dir` or the relative path "./dataset" to access the data files.',
            f"{str(self.data_preview)}",
            'You have to save the predictions result on testing set in "/content/submission.csv".',
            'Note that the testing file DOES NOT have the target column.'
        ]
        system_message = system_prompt
        user_message = "\n".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        system_prompt = "You are an AI assistant."
        user_prompt = [
            f"Task description: {str(self.cfg.task_goal)} "
            f"Memory: {str(self.journal.generate_summary())} "
            f"Previous solution: Code: {str(wrap_code(parent_node.code))} "
        ]
        system_message = system_prompt
        user_message = " ".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        return Node(plan=plan, code=code, parent=parent_node)

    def _debug(self, parent_node: Node) -> Node:
        system_prompt = "You are an AI agent."
        user_prompt = [
            f"Task description: {str(self.cfg.task_goal)}\n\n",
            f"Previous (buggy) implementation: {str(wrap_code(parent_node.code))}\n\n",
            f"Execution output: {str(wrap_code(parent_node.term_out, lang=''))}\n\n",
            str(self.data_preview)
        ]
        system_message = system_prompt
        user_message = " ".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(self):
        self.data_preview = data_preview_generate(self.cfg.data_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        parent_node = self.search_policy()
        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(node=result_node, exec_result=exec_callback(result_node.code, True))
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        node.absorb_exec_result(exec_result)
        system_prompt = "You are an AI assistant that analyzes code execution results."
        user_prompt = f"""
            Analyze the following code, its execution output, and determine if it's buggy.
            The task is: {self.cfg.task_goal}

            The code implementation is:
            {wrap_code(node.code)}

            The execution output is:
            {wrap_code(node.term_out, lang="")}

            Based on the analysis, provide your evaluation in the following JSON format.
            Do not include any other text outside of the JSON object.

            JSON schema:
            {{
                "is_buggy": <true if the code has bugs or failed, otherwise false>,
                "metric": <the calculated Mean Squared Error (MSE) as a float, or null if not available>,
                "summary": "<a brief summary of the execution results and feedback for improvement>"
            }}
        """

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        try:
            extracted_json = extract_jsons(response)
            if not extracted_json:
                raise ValueError("No JSON object found in the LLM response.")
            evaluation = Evaluation.model_validate(extracted_json[0])
            node.analysis = evaluation.summary
            node.metric = evaluation.metric
            node.is_buggy = (
                evaluation.is_buggy
                or node.exc_type is not None
                or evaluation.metric is None
            )
        except Exception as e:
            print(f"Failed to parse LLM evaluation response: {e}")
            print(f"Raw response: {response}")
            node.analysis = f"Error during evaluation: {e}"
            node.is_buggy = True
            node.metric = None


