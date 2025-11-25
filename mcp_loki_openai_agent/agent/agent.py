from mcp_context.context import MCPContext
from analyzer.fortran_analyzer import FortranAnalyzer
from analyzer.loki_integration import LokiIntegration
from agent.ia_interface import OpenAIInterface
import os

class Agent:
    def __init__(self, fortran_file):
        self.context = MCPContext()
        self.analyzer = FortranAnalyzer(fortran_file)
        self.loki = LokiIntegration(fortran_file, self.context)
        self.llm = OpenAIInterface()
        self.fortran_file = fortran_file
        self.lines = self.analyzer.lines

    def analyze(self):
        for ln, _ in self.analyzer.detect_long_lines():
            self.context.add_issue(self.fortran_file, f"{ln}-{ln}", "long_line", status="pending")
        for start, end in self.analyzer.detect_nested_loops():
            self.context.add_issue(self.fortran_file, f"{start}-{end}", "nested_loop", status="pending")

    def propose(self):
        for idx, issue in self.context.get_pending():
            start, end = map(int, issue["line_range"].split("-"))
            snippet = "".join(self.lines[start-1:end])
            suggestion = self.llm.suggest_refactoring(issue, snippet)
            self.context.update_issue(idx, suggestion=suggestion)

    def transform(self):
        sf = self.loki.parse()
        for idx, issue in self.context.get_pending():
            if self.loki.apply_transformation(sf, issue):
                self.context.update_issue(idx, status="transformed")
        os.makedirs("output", exist_ok=True)
        self.loki.write_transformed(sf, "output/transformed.f90")

    def add_gpu_directives(self):
        out_file = "output/transformed.f90"
        with open(out_file, "r") as f:
            lines = f.readlines()
        for idx, issue in enumerate(self.context.context):
            if issue["debt_type"] == "nested_loop" and issue["status"] == "transformed":
                start, _ = map(int, issue["line_range"].split("-"))
                directive = self.llm.suggest_gpu_directive(issue)
                lines.insert(start-1, directive + "\n")
                self.context.update_issue(idx, status="gpu_ready", suggestion=directive)
        with open("output/transformed_gpu.f90", "w") as f:
            f.writelines(lines)

    def run(self):
        self.analyze()
        self.propose()
        self.transform()
        self.add_gpu_directives()
        print("Workflow completed. MCP context:")
        self.context.show()

if __name__ == "__main__":
    agent = Agent("examples/solver.f90")
    agent.run()
