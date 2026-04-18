import argparse
import sys
from local_code_agent.agent.translation_graph import translation_app, performance_agent
from local_code_agent.agent.translation_graph_phase1 import translation_app_phase1


def _read_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)


def translate_file(filepath: str):
    """Phase 2 — pipeline JAX (legacy)."""
    print(f"--- JAX Translation Pipeline: {filepath} ---")
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "isolated_kernel": "",
        "jax_code": "",
        "compilation_error": "",
        "test_results": {},
        "performance_metrics": {},
    }
    final_state = translation_app.invoke(initial_state)
    output_path = filepath.replace(".f90", "_jax.py").replace(".F90", "_jax.py")
    with open(output_path, "w") as f:
        f.write(final_state.get("jax_code", ""))
    print(f"Translation complete → {output_path}")


def translate_file_gpu(filepath: str):
    """Phase 1 — pipeline Fortran GPU + Cython."""
    print(f"--- GPU Translation Pipeline (Phase 1): {filepath} ---")
    code = _read_file(filepath)
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "kernel_results": [],
        "schema": {},
        "is_program": False,
        "module_fortran": "",
        "driver_fortran": "",
        "kernel_names": [],
        "pure_elemental_fortran": "",
        "openacc_fortran": "",
        "cython_pyx": "",
        "cython_header": "",
        "cython_setup": "",
        "validation_passed": False,
        "validation_log": "",
        "executed_agents": [],
    }
    final_state = translation_app_phase1.invoke(initial_state)
    status = "PASSED" if final_state.get("validation_passed") else "FAILED"
    print(f"\n--- Phase 1 Complete ---")
    print(f"Validation   : {status}")
    print(f"Output       : output/fortran_gpu/  +  output/cython/")
    if final_state.get("validation_log"):
        print(f"Validation log:\n{final_state['validation_log']}")


def profile_file(filepath: str):
    print(f"--- Performance Profile: {filepath} ---")
    state = {"fortran_filepath": filepath, "performance_metrics": {}}
    state = performance_agent(state)  # type: ignore
    print("Performance Results:", state["performance_metrics"])


# ── UV Entry Points ────────────────────────────────────────────────────────────

def run_translate():
    """agent-translate — Phase 2 : Fortran → JAX"""
    parser = argparse.ArgumentParser(description="Fortran → JAX Translation (Phase 2)")
    parser.add_argument("filepath", help="Path to the .f90 file")
    args = parser.parse_args()
    translate_file(args.filepath)


def run_translate_gpu():
    """agent-gpu — Phase 1 : Fortran → Fortran GPU + Cython

    Accepts both:
      agent-gpu /path/to/kernel.f90
      agent-gpu translate /path/to/kernel.f90
    """
    parser = argparse.ArgumentParser(description="Fortran → Fortran GPU + Cython (Phase 1)")
    parser.add_argument("args", nargs="+", help="[translate] <filepath.f90>")
    parsed = parser.parse_args()
    # Strip optional subcommand name for parity with agent-pipeline syntax
    parts = [p for p in parsed.args if p not in {"translate", "profile"}]
    if not parts:
        parser.error("filepath is required")
    translate_file_gpu(parts[0])


def run_profile():
    """agent-profile — Performance benchmarking"""
    parser = argparse.ArgumentParser(description="Performance Profile Agent")
    parser.add_argument("filepath", help="Path to the .f90 file")
    args = parser.parse_args()
    profile_file(args.filepath)


def main():
    """agent-pipeline — Master dispatcher"""
    parser = argparse.ArgumentParser(description="Fortran Agent Pipeline CLI")
    parser.add_argument(
        "action",
        choices=["translate", "translate-gpu", "profile"],
        help=(
            "translate      : Fortran → JAX (Phase 2)\n"
            "translate-gpu  : Fortran → Fortran GPU + Cython (Phase 1)\n"
            "profile        : Performance benchmark"
        ),
    )
    parser.add_argument("filepath", help="Path to the .f90 file")
    args = parser.parse_args()

    if args.action == "translate":
        translate_file(args.filepath)
    elif args.action == "translate-gpu":
        translate_file_gpu(args.filepath)
    elif args.action == "profile":
        profile_file(args.filepath)


if __name__ == "__main__":
    main()
