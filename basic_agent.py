# agent_base.py

class ContextMemory:
    def __init__(self):
        self.entries = []

    def add(self, agent_name, step, output):
        self.entries.append(f"[{agent_name} - {step}]\n{output}")

    def get_context(self):
        return "\n\n".join(self.entries)

class BasicAgent:
    def __init__(self, name="Agent"):
        self.name = name
        self.memory = []
        self.tools = {}
        self.task = None

    def set_task(self, task_description: str):
        self.task = task_description
        print(f"[{self.name}] Task received: {task_description}")

    def register_tool(self, name, function):
        self.tools[name] = function
        print(f"[{self.name}] Tool registered: {name}")

    def perceive(self, input_data):
        """Ingest and possibly interpret input."""
        print(f"[{self.name}] Perceived input: {input_data}")
        return input_data

    def reason(self, input_data):
        """Chain-of-thought reasoning to plan actions."""
        print(f"[{self.name}] Reasoning...")
        steps = [
            "Understand the task",
            "Identify relevant tools",
            "Plan sequence of actions"
        ]
        return steps

    def act(self, plan):
        """Execute the steps using tools."""
        print(f"[{self.name}] Executing plan...")
        for step in plan:
            print(f" - {step}")
        if "tool_name" in self.tools:
            return self.tools["tool_name"]()
        else:
            print("No tool to execute.")

    def reflect(self, outcome):
        """Self-evaluate and optionally retry or update memory."""
        print(f"[{self.name}] Reflection on outcome: {outcome}")
        self.memory.append(outcome)
        # Simple feedback logic
        if "error" in outcome.lower():
            print(f"[{self.name}] Something went wrong. Retrying...")

    def run(self, input_data):
        """Main loop."""
        perception = self.perceive(input_data)
        plan = self.reason(perception)
        result = self.act(plan)
        self.reflect(str(result))

