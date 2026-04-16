import argparse
import sys
from local_code_agent.agent.translation_graph import translation_app, performance_agent

def translate_file(filepath: str):
    print(f"--- Starting Translation Pipeline for {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
        
    initial_state = {
        "fortran_filepath": filepath,
        "fortran_code": code,
        "ast_info": {},
        "isolated_kernel": "",
        "jax_code": "",
        "compilation_error": "",
        "test_results": {},
        "performance_metrics": {}
    }
    final_state = translation_app.invoke(initial_state)
    
    # Save output JAX file
    output_path = filepath.replace(".f90", "_jax.py").replace(".F90", "_jax.py")
    with open(output_path, "w") as f:
        f.write(final_state["jax_code"])
    print(f"\nTranslation Complete. Written to -> {output_path}")

def profile_file(filepath: str):
    print(f"--- Running Performance Profile for {filepath} ---")
    state = {"fortran_filepath": filepath, "performance_metrics": {}}
    state = performance_agent(state)  # type: ignore
    print("Performance Results:", state["performance_metrics"])

def run_translate():
    """UV Entry point for `agent-translate`"""
    parser = argparse.ArgumentParser(description="Translation Agent CLI")
    parser.add_argument("filepath", type=str, help="Path to the Fortran file")
    args = parser.parse_args()
    translate_file(args.filepath)

def run_profile():
    """UV Entry point for `agent-profile`"""
    parser = argparse.ArgumentParser(description="Performance Profile Agent CLI")
    parser.add_argument("filepath", type=str, help="Path to the Fortran file")
    args = parser.parse_args()
    profile_file(args.filepath)

def main():
    """UV Entry point for the master dispatcher `agent-pipeline`"""
    parser = argparse.ArgumentParser(description="CLI for Fortran-to-JAX Agents")
    parser.add_argument("action", choices=["translate", "profile"], help="Action to perform")
    parser.add_argument("filepath", type=str, help="Path to the Fortran file")
    args = parser.parse_args()

    if args.action == "translate":
        translate_file(args.filepath)
    elif args.action == "profile":
        profile_file(args.filepath)

if __name__ == "__main__":
    main()
