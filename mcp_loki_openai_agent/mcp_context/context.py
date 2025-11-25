import json
from pathlib import Path

class MCPContext:
    def __init__(self, context_file="mcp_context/storage.json"):
        self.context_file = Path(context_file)
        if not self.context_file.exists():
            self.context = []
            self._save()
        else:
            with open(self.context_file, "r") as f:
                self.context = json.load(f)

    def add_issue(self, file, line_range, debt_type, suggestion=None, status="pending", iteration=0):
        issue = {
            "file": file,
            "line_range": line_range,
            "debt_type": debt_type,
            "suggestion": suggestion,
            "status": status,
            "iteration": iteration
        }
        self.context.append(issue)
        self._save()

    def update_issue(self, index, **kwargs):
        for k, v in kwargs.items():
            if k in self.context[index]:
                self.context[index][k] = v
        self._save()

    def get_pending(self):
        return [(i, issue) for i, issue in enumerate(self.context) if issue["status"] == "pending"]

    def show(self):
        for i, issue in enumerate(self.context):
            print(i, issue)
