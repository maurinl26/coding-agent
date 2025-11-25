import re

class FortranAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.lines = f.readlines()

    def detect_long_lines(self, max_len=120):
        issues = []
        for i, line in enumerate(self.lines):
            if len(line) > max_len:
                issues.append((i + 1, line.rstrip()))
        return issues

    def detect_nested_loops(self):
        issues = []
        loop_stack = []
        for i, line in enumerate(self.lines):
            stripped = line.strip().lower()
            if re.match(r"do\s+\w*\s*=", stripped):
                loop_stack.append(i + 1)
            if re.match(r"end\s+do", stripped) and loop_stack:
                if len(loop_stack) > 1:
                    issues.append((loop_stack[-2], i + 1))
                loop_stack.pop()
        return issues
