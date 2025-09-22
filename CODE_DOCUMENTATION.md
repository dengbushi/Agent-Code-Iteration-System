# 代码文档 - 函数与类功能说明

本文档按功能分类详细说明项目中每个函数和类的作用。

## 🎯 核心代理系统

### Agent类 (`agent.py`)
**主要职责**: AI代理的核心决策引擎，负责整个代码生成和优化流程的控制。

#### 核心方法

**`__init__(cfg, journal, llm)`**
- 初始化代理，设置配置、状态管理器和LLM客户端

**`step(exec_callback)`** 🚀
- **核心执行方法**：代理的单步执行逻辑
- 根据搜索策略选择下一个操作（生成草稿/调试/改进）
- 执行代码并解析结果，更新状态

```python
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
```

**`search_policy()`** 🧠
- **智能搜索策略**：决定下一步应该执行什么操作
- 基于概率选择调试有问题的节点或改进最佳节点
- 实现探索与利用的平衡

```python
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
```

#### 代码生成方法

**`_draft()`**
- 生成初始解决方案草稿
- 基于任务描述和数据预览创建第一版代码

```python
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
```

**`_debug(parent_node)`**
- 调试有问题的代码
- 基于错误信息和执行输出修复代码

```python
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
```

**`_improve(parent_node)`**
- 改进现有的良好解决方案
- 基于历史记录和当前最佳方案进行优化

```python
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
```

#### 辅助方法

**`plan_and_code_query(system_message, user_message)`**
- 与LLM交互生成计划和代码
- 处理重试逻辑和代码提取

```python
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
```

**`parse_exec_result(node, exec_result)`**
- 解析代码执行结果
- 使用LLM评估代码质量和性能指标

```python
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
```

**`update_data_preview()`**
- 更新数据预览信息，为代码生成提供上下文

```python
def update_data_preview(self):
    self.data_preview = data_preview_generate(self.cfg.data_dir)
```

### Evaluation类 (`agent.py`)
**数据模型**: 代码执行结果的评估结构

```python
class Evaluation(BaseModel):
    is_buggy: bool = Field(...)
    metric: float | None = Field(...)
    summary: str = Field(...)
```

- `is_buggy`: 代码是否有错误
- `metric`: 性能指标（MSE）
- `summary`: 执行结果摘要

## 📊 状态管理系统

### Node类 (`models.py`)
**主要职责**: 表示解决方案树中的单个节点，包含完整的解决方案信息。

#### 核心属性
- `code`: 解决方案的Python代码
- `plan`: 自然语言描述的解决计划
- `parent/children`: 树状结构的父子关系
- `step`: 在整个搜索过程中的步骤编号

#### 执行信息
- `_term_out`: 代码执行的输出信息
- `exec_time`: 代码执行时间
- `exc_type/exc_info/exc_stack`: 异常信息

#### 评估信息
- `analysis`: LLM对执行结果的分析
- `metric`: 性能指标（MSE值）
- `is_buggy`: 是否存在错误

#### 重要方法

**`absorb_exec_result(exec_result)`**
- 吸收代码执行结果，更新节点状态

```python
def absorb_exec_result(self, exec_result: ExecutionResult):
    self._term_out = exec_result.term_out
    self.exec_time = exec_result.exec_time
    self.exc_type = exec_result.exc_type
    self.exc_info = exec_result.exc_info
    self.exc_stack = exec_result.exc_stack
```

**`stage_name`** (属性)
- 返回节点阶段：`draft`（草稿）、`debug`（调试）、`improve`（改进）

```python
@property
def stage_name(self) -> Literal["draft", "debug", "improve"]:
    if self.parent is None:
        return "draft"
    return "debug" if self.parent.is_buggy else "improve"
```

**`is_leaf`** (属性)
- 判断是否为叶子节点（无子节点）

```python
@property
def is_leaf(self) -> bool:
    return not self.children
```

**`debug_depth`** (属性)
- 计算调试深度（连续调试的次数）

```python
@property
def debug_depth(self) -> int:
    if self.stage_name != "debug":
        return 0
    return self.parent.debug_depth + 1  # type: ignore
```

### Journal类 (`models.py`)
**主要职责**: 管理整个解决方案搜索过程的状态和历史记录。

#### 核心方法

**`append(node)`**
- 添加新节点到搜索历史
- 自动设置步骤编号

```python
def append(self, node: Node) -> None:
    node.step = len(self.nodes)
    self.nodes.append(node)
```

**`get_best_node(only_good=True)`** 🏆
- 获取性能最佳的节点
- 可选择只考虑无错误的节点

```python
def get_best_node(self, only_good=True) -> Optional[Node]:
    if only_good:
        nodes = self.good_nodes
        if not nodes:
            return None
    else:
        nodes = self.nodes
    return min(nodes, key=lambda n: n.metric)
```

#### 节点分类属性

**`draft_nodes`** (属性)
- 返回所有草稿节点（无父节点的根节点）

```python
@property
def draft_nodes(self) -> list[Node]:
    return [n for n in self.nodes if n.parent is None]
```

**`buggy_nodes`** (属性)
- 返回所有有错误的节点

```python
@property
def buggy_nodes(self) -> list[Node]:
    return [n for n in self.nodes if n.is_buggy]
```

**`good_nodes`** (属性)
- 返回所有无错误的节点

```python
@property
def good_nodes(self) -> list[Node]:
    return [n for n in self.nodes if not n.is_buggy]
```

#### 分析方法

**`generate_summary(include_code=False)`**
- 生成搜索过程的摘要报告
- 包含设计思路、结果分析和性能指标

```python
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
```

**`get_metric_history()`**
- 获取所有节点的性能指标历史

```python
def get_metric_history(self) -> list[float]:
    return [n.metric for n in self.nodes]
```

## ⚡ 代码执行引擎

### Interpreter类 (`interpreter_module.py`)
**主要职责**: 提供安全的、隔离的Python代码执行环境。

#### 核心方法

**`run(code, reset_session=True)`** 🔥
- **主要执行方法**：在隔离进程中执行Python代码
- 支持会话重置和持续会话
- 返回完整的执行结果

```python
def run(self, code: str, reset_session=True) -> ExecutionResult:
    if reset_session:
        if self.process is not None:
            self.cleanup_session()
        self.create_process()
    else:
        assert self.process is not None

    assert self.process.is_alive()
    self.code_inq.put(code)

    try:
        state = self.event_outq.get(timeout=30)
    except queue.Empty:
        msg = "REPL child process failed to start execution"
        while not self.result_outq.empty():
            break
        raise RuntimeError(msg) from None
    assert state[0] == "state:ready", state
    start_time = time.time()

    child_in_overtime = False
    while True:
        try:
            state = self.event_outq.get(timeout=1)
            assert state[0] == "state:finished", state
            exec_time = time.time() - start_time
            break
        except queue.Empty:
            if time.time() - start_time > self.timeout:
                if not child_in_overtime:
                    child_in_overtime = True
                    self.process.terminate()
                else:
                    self.process.kill()
                    break
            continue

    # 收集输出结果
    term_out = []
    while not self.result_outq.empty():
        try:
            msg = self.result_outq.get_nowait()
            if msg == "<|EOF|>":
                break
            term_out.append(msg)
        except queue.Empty:
            break

    return ExecutionResult(
        term_out=term_out,
        exec_time=exec_time,
        exc_type=state[1] if len(state) > 1 else None,
        exc_info=state[2] if len(state) > 2 else None,
        exc_stack=state[3] if len(state) > 3 else None,
    )
```

**`create_process()`**
- 创建新的子进程用于代码执行
- 设置进程间通信队列

```python
def create_process(self) -> None:
    ctx = mp.get_context("spawn")
    self.code_inq, self.result_outq, self.event_outq = ctx.Queue(), ctx.Queue(), ctx.Queue()
    self.process = ctx.Process(target=self._run_session, args=(self.code_inq, self.result_outq, self.event_outq))
    self.process.daemon = True
    self.process.start()
```

**`cleanup_session()`**
- 清理执行进程，释放资源
- 处理进程终止和强制杀死

```python
def cleanup_session(self):
    if self.process is None:
        return
    try:
        self.process.terminate()
        self.process.join(timeout=0.5)
        if self.process.exitcode is None:
            self.process.kill()
            self.process.join(timeout=0.5)
            if self.process.exitcode is None:
                os.kill(self.process.pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error during process cleanup: {e}")
    finally:
        if self.process is not None:
            self.process.close()
            self.process = None
```

#### 内部方法

**`_run_session(code_inq, result_outq, event_outq)`**
- 子进程中的主执行循环
- 处理代码编译和执行
- 捕获异常和输出

**`child_proc_setup(result_outq)`**
- 子进程初始化设置
- 重定向标准输出和错误输出

### ExecutionResult类 (`interpreter_module.py`)
**数据模型**: 代码执行结果的完整信息
- `term_out`: 终端输出信息
- `exec_time`: 执行时间
- `exc_type`: 异常类型
- `exc_info`: 异常详细信息
- `exc_stack`: 异常堆栈

### RedirectQueue类 (`interpreter_module.py`)
**辅助类**: 将标准输出重定向到队列
- `write(msg)`: 写入消息到队列
- `flush()`: 刷新缓冲区（空实现）

## 🤖 LLM交互系统

### LLMClient类 (`llm_client.py`)
**主要职责**: 封装与大语言模型的交互接口。

**`__init__(api_key, base_url)`**
- 初始化OpenAI客户端连接

**`generate_response(messages, model, temperature, max_tokens, stop)`** 💬
- **核心方法**：生成LLM响应
- 支持多轮对话和参数配置
- 处理API错误和异常情况

```python
def generate_response(self, messages: list[dict],
                      model: str = "deepseek-chat",
                      temperature: float = 0.0,
                      max_tokens: int = 4096,
                      stop: list[str] | None = None) -> str:
    if stop is None:
        stop = ["<|eot_id|>", "<|end_of_text|>"]
    try:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"
```

## 📝 文本处理工具

### 代码处理函数 (`text_utils.py`)

**`extract_code(text)`** 🔍
- **关键函数**：从LLM响应中提取Python代码块
- 支持多种代码块格式
- 验证代码语法正确性

```python
def extract_code(text: str) -> str:
    parsed_codes: list[str] = []
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)
    valid_code_blocks = [c for c in parsed_codes if is_valid_python_script(c)]
    return "\n\n".join(valid_code_blocks)
```

**`extract_text_up_to_code(s)`**
- 提取代码块之前的文本内容（通常是计划描述）

```python
def extract_text_up_to_code(s: str) -> str:
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()
```

**`wrap_code(code, lang="python")`**
- 将代码包装成Markdown格式的代码块

```python
def wrap_code(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code}\n```"
```

**`is_valid_python_script(script)`**
- 验证Python代码的语法正确性

```python
def is_valid_python_script(script: str) -> bool:
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False
```

### JSON处理函数 (`text_utils.py`)

**`extract_jsons(text)`**
- 从文本中提取所有有效的JSON对象
- 用于解析LLM返回的结构化评估结果

```python
def extract_jsons(text: str) -> list[dict]:
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass
    return json_objects
```

### 文本工具函数 (`text_utils.py`)

**`trim_long_string(string, threshold=5100, k=2500)`**
- 截断过长的字符串，保留首尾部分
- 防止输出信息过长影响性能

```python
def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    return string
```

## 🛠️ 通用工具函数

### 数据处理函数 (`utils.py`)

**`data_preview_generate(base_path)`** 📊
- **重要函数**：生成数据集预览信息
- 为代理提供数据上下文，指导代码生成

```python
def data_preview_generate(base_path):
    result = []
    files = [p for p in Path(base_path).iterdir()]
    for f in sorted(files):
        result.append(preview_csv(f))
    return "\n\n".join(result)
```

**`preview_csv(p)`**
- 生成单个CSV文件的详细预览
- 包含数据形状、列信息、统计摘要等

```python
def preview_csv(p: Path) -> str:
    df = pd.read_csv(p)
    out: list[str] = []
    out.append(f"--- Data Preview for {str(p)} ---")
    out.append(f"Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    out.append("-" * 20)
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=False)
    info_str = buffer.getvalue()
    out.append("Column Overview (Name, Non-Null Count, Dtype):")
    out.append(info_str)
    out.append("-" * 20)
    out.append("Data Head:")
    out.append(df.head().to_string())
    out.append("-" * 20)
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        out.append("Numerical Columns Statistics:")
        out.append(numeric_df.describe().to_string())
        out.append("-" * 20)
    if 'sample_submission.csv' in str(p):
        out.append("Submission File Format:")
        out.append("The goal is to predict the 'positive' column for each 'id'.")
    out.append(f"--- End of Preview for {str(p)} ---")
    return "\n\n".join(out)
```

### 结果保存函数 (`utils.py`)

**`save_run(cfg, journal)`**
- 保存搜索过程的最佳解决方案
- 生成 `best_solution.py` 和多个 `good_solution_*.py` 文件

```python
def save_run(cfg, journal):
    best_node = journal.get_best_node(only_good=False)
    if best_node is not None:
        with open("best_solution.py", "w", encoding="utf-8") as f:
            f.write(best_node.code)

    good_nodes = journal.get_good_nodes()
    for i, node in enumerate(good_nodes):
        filename = f"good_solution_{i}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(node.code)
```

## 🔧 配置系统

### Config类 (`config.py`)
**配置管理**: 递归地将字典转换为对象属性访问

```python
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
```

**`__init__(dictionary)`**
- 支持嵌套字典的属性化访问

### 配置函数 (`config.py`)

**`set_seed(seed=531)`**
- 设置随机种子，确保实验可重现

```python
def set_seed(seed=531):
    random.seed(seed)
    np.random.seed(seed)
```

## 🚀 程序入口

### main函数 (`main.py`)
**主程序**: 协调所有组件，执行完整的代理搜索流程

```python
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
```

执行流程：
1. 加载环境变量和API密钥
2. 初始化Journal、LLMClient、Agent和Interpreter
3. 执行指定步数的代理搜索
4. 保存最终结果

## 📋 功能流程总结

### 完整执行流程
1. **初始化** → 创建代理、状态管理器、执行引擎
2. **数据预览** → 生成数据集上下文信息
3. **搜索循环** → 重复执行以下步骤：
   - 搜索策略决策
   - 代码生成（草稿/调试/改进）
   - 代码执行
   - 结果评估
   - 状态更新
4. **结果保存** → 输出最佳解决方案

### 关键设计模式
- **策略模式**: 搜索策略的动态选择
- **状态模式**: 节点阶段的管理
- **观察者模式**: 执行结果的处理
- **工厂模式**: 节点的创建和管理

这个架构实现了AIDE论文中提出的AI驱动代码空间探索的核心理念，通过树搜索和智能重用实现了自动化的机器学习工程。
