import json
import re


def wrap_code(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script: str) -> bool:
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


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


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    return string


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


def extract_text_up_to_code(s: str) -> str:
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


