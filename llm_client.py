from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

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


