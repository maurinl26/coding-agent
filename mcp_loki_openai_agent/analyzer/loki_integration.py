from loki.frontend import Frontend
from loki import ir, backend, transformations
from mcp_context.context import MCPContext

class LokiIntegration:
    def __init__(self, fortran_file: str, mcp: MCPContext):
        self.fortran_file = fortran_file
        self.mcp = mcp

    def parse(self):
        return ir.Sourcefile.from_file(self.fortran_file, frontend=Frontend(fparser=True))

    def apply_transformation(self, sf, issue):
        debt = issue["debt_type"]
        trafo = None
        if debt == "nested_loop":
            trafo = transformations.SCCTransformation(horizontal_iter="i", hoist_allocations=True)
        elif debt == "common_block":
            trafo = transformations.RemoveCommonBlocksTransformation()
        if trafo:
            trafo.apply(sf)
            return True
        return False

    def write_transformed(self, sf, out_path: str):
        backend.FortranWriter().run(sf, out_path)
        print(f"Transformed code written to {out_path}")
