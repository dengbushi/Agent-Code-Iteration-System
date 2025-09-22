import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin

from interpreter_module import ExecutionResult
from text_utils import trim_long_string


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # general attrs
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # execution info
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # evaluation
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: float = field(default=None, kw_only=True)  # type: ignore
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class Journal(DataClassJsonMixin):
    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[float]:
        return [n.metric for n in self.nodes]

    def get_good_nodes(self) -> list[Node]:
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good=True) -> Optional[Node]:
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return min(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric (Mean Squared Error): {n.metric}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)


