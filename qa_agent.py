import openai
from basic_agent import BasicAgent


class QAAgent(BasicAgent):
    def __init__(self, name: str, model: str = "gpt-4"):
        super().__init__(name)
        self.model = model

    def run(self, question: str, context: str) -> str:
        print(f"[{self.name}] Asking ChatGPT: {question}")

        system_prompt = "You are a medical question-answering assistant. Use the context to answer user questions precisely."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during QA: {str(e)}"

