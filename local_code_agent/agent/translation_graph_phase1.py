"""
Graph d'agents LangGraph — Phase 1 : Fortran → Fortran GPU + Cython.

Pipeline :
  init → parser → extractor → pure_elemental → openacc → cython_wrapper → validation → END

Étape clé — extractor :
  Les codes scientifiques comme seismic_CPML sont des PROGRAM monolithiques avec des
  boucles compute inlines. L'extractor identifie ces boucles 2D et les extrait en
  subroutines dans un MODULE Fortran avec des INTENT explicites. Le PROGRAM devient
  un driver qui appelle le MODULE. Cela permet ensuite d'annoter chaque kernel
  séparément avec PURE/ELEMENTAL et OpenACC.

Compilateur cible : nvfortran (NVIDIA HPC SDK) — flags : -acc -gpu=cc80 (A100 Ampere)
Interface Python  : Cython + NumPy typed memoryviews (np.float64_t[:,:])
"""

import sys
import os
import re
import operator
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, END
from local_code_agent.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


SEP  = "─" * 64
SEP2 = "═" * 64


# ==========================================
# 1. État du Graphe
# ==========================================

class KernelInfo(TypedDict):
    routine_name: str
    fortran_code: str           # Code source Fortran original (extrait par Loki)
    pure_elemental_code: str    # Code annoté PURE/ELEMENTAL (ou original si non éligible)
    openacc_code: str           # Code avec pragmas OpenACC
    intent_map: Dict[str, str]  # {arg_name: "IN"|"OUT"|"INOUT"}
    is_pure: bool
    is_elemental: bool
    has_io: bool                # I/O Fortran (PRINT/WRITE/READ) détecté par Loki
    has_save: bool              # Variables SAVE détectées par Loki
    loops: List[str]            # Descriptions des bornes de boucles
    dimensions: Dict[str, Any]  # {var_name: [dim1, dim2, ...]}
    status: str                 # "pending" | "success" | "error"
    error_log: str


class Phase1State(TypedDict):
    fortran_filepath: str
    fortran_code: str
    # Loki AST analysis
    ast_info: Dict[str, Any]
    kernel_results: List[KernelInfo]   # Plain list — replaced (not appended) at each step
    schema: Dict[str, Any]             # {params, statics, state}
    is_program: bool
    # Extractor outputs (produced before PURE/ELEMENTAL)
    module_fortran: str     # MODULE contenant les kernels extraits (module_kernels.f90)
    driver_fortran: str     # PROGRAM driver appelant le MODULE (driver.f90)
    kernel_names: List[str] # Noms des subroutines extraites
    # Phase outputs
    pure_elemental_fortran: str        # Fortran annoté PURE/ELEMENTAL
    openacc_fortran: str               # Fortran avec pragmas OpenACC (kernels + driver data region)
    cython_pyx: str                    # Contenu .pyx
    cython_header: str                 # Contenu kernel_c.h (iso_c_binding)
    cython_setup: str                  # pyproject.toml build config
    # Validation
    validation_passed: bool
    validation_log: str
    # Tracking
    executed_agents: List[str]


# ==========================================
# 2. Helpers
# ==========================================

def _out(category: str = "fortran_gpu") -> Path:
    p = Path("output") / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(path: Path, content: str):
    path.write_text(content, encoding="utf-8")
    print(f"  Saved → {path}")


def _strip_markdown(code: str) -> str:
    match = re.search(r"```[a-zA-Z0-9]*\n?(.*?)\n?```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def _gpu_compiler() -> str | None:
    """Retourne le premier compilateur Fortran GPU trouvé (nvfortran ou pgfortran)."""
    for compiler in ["nvfortran", "pgfortran"]:
        if shutil.which(compiler):
            return compiler
    env_fc = os.getenv("FC")
    if env_fc and shutil.which(env_fc):
        return env_fc
    return None


# ==========================================
# 3. Agents (Nodes)
# ==========================================

# ── Node 0 : Init ────────────────────────────────────────────────────────────

def init_phase1(state: Phase1State) -> dict:
    """Crée la structure de sortie et vérifie la présence du compilateur GPU."""
    print(f"\n{SEP}")
    print("  [Init] Fortran GPU Phase 1")
    print(SEP)

    for d in ["fortran_gpu", "cython"]:
        _out(d)
        print(f"  Dir : output/{d}/")

    compiler = _gpu_compiler()
    if compiler:
        print(f"  GPU compiler : {compiler}")
    else:
        print("  WARNING: nvfortran/pgfortran not found. Set FC env var or install NVIDIA HPC SDK.")
        print("           Compilation step will be skipped in validation.")

    print(SEP + "\n")
    return {}


# ── Node 1 : Parser (Loki AST) ───────────────────────────────────────────────

def parser_phase1(state: Phase1State) -> dict:
    """Parse le fichier Fortran avec Loki, extrait routines + schéma global.

    Fixe le bug UnboundLocalError de l'agent original : sys est importé au niveau
    module et n'est jamais réimporté à l'intérieur du corps de la fonction.
    """
    filepath = state["fortran_filepath"]
    print(f"\n{SEP}")
    print(f"  [Parser] Loki AST — {filepath}")
    print(SEP)

    # Injection du fork local Loki (PROGRAM support patches)
    _loki_local = str(Path(__file__).resolve().parents[2] / "loki")
    if _loki_local not in sys.path:
        sys.path.insert(0, _loki_local)

    try:
        from loki import Sourcefile, Frontend, FindNodes, BasicType
        from loki.ir.nodes import VariableDeclaration, Loop, CallStatement

        raw_content = Path(filepath).read_text(encoding="utf-8")
        is_program = bool(re.search(r'^\s*PROGRAM\s+', raw_content, re.IGNORECASE | re.MULTILINE))

        if is_program:
            print("  PROGRAM block detected — converting to SUBROUTINE for Loki.")
            raw_content = re.sub(
                r'^\s*PROGRAM\s+(\w+)', r'SUBROUTINE \1_master',
                raw_content, flags=re.IGNORECASE | re.MULTILINE
            )
            raw_content = re.sub(
                r'^\s*END\s+PROGRAM', r'END SUBROUTINE',
                raw_content, flags=re.IGNORECASE | re.MULTILINE
            )

        fd, tmp_path = tempfile.mkstemp(suffix='.f90')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(raw_content)

            # Safety check via subprocess (isolates fparser crashes)
            check = subprocess.run(
                [sys.executable, "-c",
                 f"from loki import Sourcefile; Sourcefile.from_file(r'{tmp_path}')"],
                capture_output=True, timeout=30
            )
            if check.returncode == 0:
                source = Sourcefile.from_file(tmp_path)
            else:
                print(f"  FParser subprocess failed (rc={check.returncode}) — using REGEX frontend.")
                source = Sourcefile.from_file(tmp_path, frontend=Frontend.REGEX)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not source.routines:
            raise ValueError("Loki found no routines in this file.")

        print(f"  Parsed {len(source.routines)} routine(s).")

        # ── Global schema extraction ──────────────────────────────────────────
        all_params, all_statics, all_state = [], [], []
        for routine in source.routines:
            for decl in FindNodes(VariableDeclaration).visit(routine.spec):
                for var in decl.symbols:
                    is_param = getattr(var.type, 'parameter', False)
                    dtype    = getattr(var.type, 'dtype', BasicType.DEFERRED)
                    if is_param:
                        (all_statics if dtype == BasicType.LOGICAL else all_params).append(var.name)
                    elif hasattr(var, 'dimensions') and var.dimensions:
                        all_state.append(var.name)

        if not all_params and not all_state:
            print("  Loki schema empty — REGEX fallback.")
            raw_text = Path(filepath).read_text(encoding="utf-8")
            all_params  = re.findall(r'(\w+)\s*,\s*parameter', raw_text, re.IGNORECASE)
            all_statics = re.findall(r'LOGICAL\s*,\s*parameter\s*::\s*(\w+)', raw_text, re.IGNORECASE)
            all_state   = re.findall(r'(\w+)\s*\([^)]+\)', raw_text)

        schema = {
            "params":  sorted(set(all_params)),
            "statics": sorted(set(all_statics)),
            "state":   sorted(set(all_state)),
        }
        print(f"  Schema: {len(schema['params'])} params | "
              f"{len(schema['statics'])} statics | {len(schema['state'])} state vars")

        # ── Per-routine analysis ──────────────────────────────────────────────
        kernel_results: List[KernelInfo] = []
        io_keywords = {'print', 'write', 'read', 'open', 'close', 'inquire'}

        for routine in source.routines:
            intent_map: Dict[str, str] = {}
            if hasattr(routine, 'arguments'):
                for v in routine.arguments:
                    intent = getattr(v.type, 'intent', None)
                    if intent:
                        intent_map[v.name] = intent.upper()

            loops = FindNodes(Loop).visit(routine.body)
            loop_descriptions = [
                str(lp.bounds) if hasattr(lp, 'bounds') else "?" for lp in loops
            ]

            has_io = any(
                any(k in str(node.name).lower() for k in io_keywords)
                for node in FindNodes(CallStatement).visit(routine.body)
            )
            if not has_io:
                fortran_str = routine.to_fortran().lower()
                has_io = any(k in fortran_str for k in io_keywords)

            has_save = False
            if hasattr(routine, 'variables'):
                has_save = any(getattr(v.type, 'save', False) for v in routine.variables)

            dimensions: Dict[str, Any] = {}
            for decl in FindNodes(VariableDeclaration).visit(routine.spec):
                for var in decl.symbols:
                    if hasattr(var, 'dimensions') and var.dimensions:
                        dimensions[var.name] = [str(d) for d in var.dimensions]

            kernel_results.append({
                "routine_name":       routine.name,
                "fortran_code":       routine.to_fortran(),
                "pure_elemental_code": "",
                "openacc_code":       "",
                "intent_map":         intent_map,
                "is_pure":            False,
                "is_elemental":       False,
                "has_io":             has_io,
                "has_save":           has_save,
                "loops":              loop_descriptions,
                "dimensions":         dimensions,
                "status":             "pending",
                "error_log":          "",
            })
            print(f"  Routine: {routine.name} | loops={len(loops)} | "
                  f"io={has_io} | save={has_save} | args={len(intent_map)}")

        return {
            "kernel_results": kernel_results,
            "schema": schema,
            "is_program": is_program,
            "ast_info": {
                "status": "parsed",
                "routines": [k["routine_name"] for k in kernel_results],
            },
            "executed_agents": list(state.get("executed_agents", [])) + ["parser"],
        }

    except Exception as e:
        import traceback
        print(f"  Loki failed: {e}")
        traceback.print_exc()
        raw = Path(filepath).read_text(encoding="utf-8")
        return {
            "kernel_results": [{
                "routine_name": "kernel",
                "fortran_code": raw,
                "pure_elemental_code": "",
                "openacc_code": "",
                "intent_map": {},
                "is_pure": False,
                "is_elemental": False,
                "has_io": False,
                "has_save": False,
                "loops": [],
                "dimensions": {},
                "status": "error",
                "error_log": str(e),
            }],
            "schema": {"params": [], "statics": [], "state": []},
            "is_program": False,
            "ast_info": {"status": "error", "message": str(e)},
            "executed_agents": list(state.get("executed_agents", [])) + ["parser"],
        }


# ── Node 2 : Extractor ───────────────────────────────────────────────────────

def extractor_agent(state: Phase1State) -> dict:
    """LLM : extrait les boucles compute du PROGRAM en subroutines dans un MODULE.

    Cas typique : codes scientifiques monolithiques (seismic_CPML) où les kernels FD
    sont des blocs do/enddo inline dans le PROGRAM. L'extraction est nécessaire pour :
      - annoter chaque kernel avec PURE/ELEMENTAL individuellement
      - ajouter !$acc parallel loop sur les boucles spatiales 2D
      - générer un wrapper Cython sur des subroutines avec INTENT explicites

    Sorties :
      module_kernels.f90 — MODULE avec N subroutines (kernels GPU purs)
      driver.f90         — PROGRAM driver appelant USE module_kernels + les subroutines
    """
    print(f"\n{SEP}")
    print("  [Extractor] Identifying and extracting compute kernels into MODULE")
    print(SEP)

    llm = get_llm()

    # Read the original source to give LLM full context
    filepath = state["fortran_filepath"]
    try:
        full_source = Path(filepath).read_text(encoding="utf-8")
    except Exception:
        full_source = state.get("fortran_code", "")

    # Truncate if very large (keep first 600 lines for context — enough for seismic_CPML)
    lines = full_source.split("\n")
    source_preview = "\n".join(lines[:700]) if len(lines) > 700 else full_source

    module_name = Path(filepath).stem.lower().replace("-", "_").replace(".", "_")

    system = SystemMessage(content=(
        "You are a Fortran HPC expert specializing in GPU refactoring.\n"
        "Your task: given a monolithic Fortran PROGRAM, extract the computational "
        "kernels (2D finite-difference loops) into subroutines inside a Fortran MODULE.\n\n"
        "Rules:\n"
        "  1. Identify all 2D spatial loop nests (do j=... / do i=...) that update "
        "     field arrays (velocities, stresses, memory variables) — these are the GPU kernels.\n"
        "  2. Each loop nest becomes ONE subroutine. Name it descriptively "
        "     (e.g. update_stress_xx_yy, update_velocity_x).\n"
        "  3. All dummy arguments must have explicit INTENT(IN), INTENT(OUT), or INTENT(INOUT):\n"
        "       - Arrays read AND written (in-place update): INTENT(INOUT)\n"
        "       - Arrays only read: INTENT(IN)\n"
        "       - Scalar parameters: INTENT(IN)\n"
        "       - Grid sizes (NX, NY): INTENT(IN), INTEGER\n"
        "  4. NO I/O (PRINT/WRITE/READ) inside the extracted subroutines.\n"
        "  5. Keep loop bounds identical to the original code.\n"
        "  6. The MODULE must use 'implicit none' and contain all subroutines.\n"
        "  7. The PROGRAM driver must:\n"
        "       - USE the module\n"
        "       - Keep all parameter declarations, PML init, material init\n"
        "       - Replace inline loop nests with calls to the MODULE subroutines\n"
        "       - Keep I/O, seismogram recording, energy computation as-is (CPU)\n\n"
        "Return TWO clearly separated code blocks:\n"
        "  [MODULE] ... [/MODULE]\n"
        "  [DRIVER] ... [/DRIVER]"
    ))

    prompt = HumanMessage(content=(
        f"Extract the GPU compute kernels from this Fortran PROGRAM into a MODULE.\n"
        f"Module name: {module_name}_kernels\n\n"
        f"Source code:\n```fortran\n{source_preview}\n```"
    ))

    try:
        resp = llm.invoke([system, prompt])
        content = resp.content

        # Parse [MODULE]...[/MODULE] and [DRIVER]...[/DRIVER] blocks
        module_match = re.search(r'\[MODULE\](.*?)\[/MODULE\]', content, re.DOTALL | re.IGNORECASE)
        driver_match = re.search(r'\[DRIVER\](.*?)\[/DRIVER\]', content, re.DOTALL | re.IGNORECASE)

        if module_match and driver_match:
            module_code = _strip_markdown(module_match.group(1).strip())
            driver_code = _strip_markdown(driver_match.group(1).strip())
        else:
            # Fallback: try to find two fortran code blocks
            blocks = re.findall(r'```fortran\n(.*?)\n```', content, re.DOTALL)
            if len(blocks) >= 2:
                module_code = blocks[0].strip()
                driver_code = blocks[1].strip()
            else:
                # Last resort: entire response is the module
                module_code = _strip_markdown(content)
                driver_code = ""
                print("  WARNING: could not parse separate MODULE/DRIVER blocks")

        # Extract kernel subroutine names from module code
        kernel_names = re.findall(r'^\s*subroutine\s+(\w+)\s*\(', module_code,
                                  re.IGNORECASE | re.MULTILINE)
        print(f"  Extracted {len(kernel_names)} kernel subroutine(s): {kernel_names}")

        # Save outputs
        _save(_out("fortran_gpu") / "module_kernels.f90", module_code)
        if driver_code:
            _save(_out("fortran_gpu") / "driver.f90", driver_code)

        # Rebuild kernel_results from extracted subroutines
        # (the parser only saw the monolithic PROGRAM; now we have real subroutines)
        updated_kernels: List[KernelInfo] = []
        for name in kernel_names:
            # Extract this subroutine's source from the module code
            sub_match = re.search(
                rf'subroutine\s+{name}\s*\(.*?end\s+subroutine\s+{name}',
                module_code, re.DOTALL | re.IGNORECASE
            )
            sub_code = sub_match.group(0) if sub_match else f"! subroutine {name} (extraction failed)"

            # Parse INTENT from extracted code
            intent_map: Dict[str, str] = {}
            for m in re.finditer(r'intent\s*\(\s*(in|out|inout)\s*\)\s*::\s*([^\n!]+)',
                                  sub_code, re.IGNORECASE):
                intent_str = m.group(1).upper()
                for var in m.group(2).replace(' ', '').split(','):
                    if var.strip():
                        intent_map[var.strip()] = intent_str

            updated_kernels.append({
                "routine_name":       name,
                "fortran_code":       sub_code,
                "pure_elemental_code": "",
                "openacc_code":       "",
                "intent_map":         intent_map,
                "is_pure":            False,
                "is_elemental":       False,
                "has_io":             False,   # extracted kernels have no I/O by construction
                "has_save":           False,
                "loops":              [],
                "dimensions":         {},
                "status":             "extracted",
                "error_log":          "",
            })

        # If extraction produced no kernels, keep the parser's kernel_results
        if not updated_kernels:
            updated_kernels = state.get("kernel_results", [])
            print("  WARNING: no subroutines extracted — keeping parser results")

        return {
            "module_fortran":  module_code,
            "driver_fortran":  driver_code,
            "kernel_names":    kernel_names,
            "kernel_results":  updated_kernels,
            "executed_agents": list(state.get("executed_agents", [])) + ["extractor"],
        }

    except Exception as e:
        import traceback
        print(f"  LLM extraction failed: {e}")
        traceback.print_exc()
        return {
            "module_fortran":  "",
            "driver_fortran":  "",
            "kernel_names":    [],
            "kernel_results":  state.get("kernel_results", []),
            "executed_agents": list(state.get("executed_agents", [])) + ["extractor"],
        }


# ── Node 3 : PURE / ELEMENTAL ────────────────────────────────────────────────

def pure_elemental_agent(state: Phase1State) -> dict:
    """LLM : annote les kernels de calcul pur avec PURE ou ELEMENTAL."""
    print(f"\n{SEP}")
    print("  [PURE/ELEMENTAL] Annotating compute kernels")
    print(SEP)

    llm = get_llm()
    system = SystemMessage(content=(
        "You are a Fortran GPU expert. Annotate Fortran subroutines with PURE or ELEMENTAL "
        "attributes where valid.\n"
        "Rules:\n"
        "  - PURE: no I/O, no SAVE, no COMMON, no hidden global state, all args must have INTENT\n"
        "  - ELEMENTAL: same as PURE and operates element-wise on scalars/arrays\n"
        "  - Only annotate pure compute kernels; skip routines with I/O, SAVE, or time loops\n"
        "  - Add explicit INTENT attributes to all dummy arguments if missing\n"
        "  - If a subroutine cannot be made PURE, return it unchanged\n"
        "Return ONLY the annotated Fortran code, no prose."
    ))

    updated: List[KernelInfo] = []
    for kernel in state.get("kernel_results", []):
        if kernel["has_io"] or kernel["has_save"]:
            print(f"  Skip {kernel['routine_name']} (io={kernel['has_io']}, save={kernel['has_save']})")
            updated.append({**kernel, "pure_elemental_code": kernel["fortran_code"]})
            continue

        prompt = HumanMessage(content=(
            f"Annotate this Fortran subroutine with PURE or ELEMENTAL if valid.\n"
            f"Loki analysis: loops={kernel['loops']}, intents={kernel['intent_map']}\n\n"
            f"```fortran\n{kernel['fortran_code']}\n```"
        ))
        try:
            resp = llm.invoke([system, prompt])
            annotated = _strip_markdown(resp.content)
            is_pure      = bool(re.search(r'\bPURE\b',      annotated, re.IGNORECASE))
            is_elemental = bool(re.search(r'\bELEMENTAL\b', annotated, re.IGNORECASE))
            print(f"  {kernel['routine_name']} → pure={is_pure}, elemental={is_elemental}")
            updated.append({
                **kernel,
                "pure_elemental_code": annotated,
                "is_pure": is_pure,
                "is_elemental": is_elemental,
            })
        except Exception as e:
            print(f"  LLM failed for {kernel['routine_name']}: {e}")
            updated.append({**kernel, "pure_elemental_code": kernel["fortran_code"], "error_log": str(e)})

    combined = "\n\n".join(k["pure_elemental_code"] for k in updated)
    _save(_out("fortran_gpu") / "kernel_pure.f90", combined)

    return {
        "kernel_results": updated,
        "pure_elemental_fortran": combined,
        "executed_agents": list(state.get("executed_agents", [])) + ["pure_elemental"],
    }


# ── Node 3 : OpenACC Insert ──────────────────────────────────────────────────

def openacc_insert_agent(state: Phase1State) -> dict:
    """Loki-informed LLM : insère les pragmas OpenACC dans les kernels Fortran."""
    print(f"\n{SEP}")
    print("  [OpenACC] Inserting OpenACC pragmas")
    print(SEP)

    llm = get_llm()
    system = SystemMessage(content=(
        "You are an OpenACC GPU expert for scientific Fortran. "
        "Insert OpenACC pragmas to parallelize the given subroutine on NVIDIA A100 GPUs.\n"
        "Compiler: nvfortran -acc -gpu=cc80 (Ampere)\n"
        "Guidelines:\n"
        "  - !$acc routine seq  for subroutines called from within a GPU parallel region\n"
        "  - !$acc parallel loop  for the outermost parallelizable loop\n"
        "  - !$acc loop vector  for inner SIMD-eligible loops\n"
        "  - !$acc data copyin(...) copyout(...)  for arrays entering/leaving GPU\n"
        "  - Use copy(...) for INTENT(INOUT) arrays\n"
        "  - !$acc end parallel  and  !$acc end data  to close regions\n"
        "  - Do NOT add OpenACC to routines that contain PRINT/WRITE/READ\n"
        "Return ONLY the annotated Fortran code."
    ))

    updated: List[KernelInfo] = []
    for kernel in state.get("kernel_results", []):
        src = kernel.get("pure_elemental_code") or kernel["fortran_code"]

        if kernel["has_io"]:
            print(f"  Skip {kernel['routine_name']} (has I/O)")
            updated.append({**kernel, "openacc_code": src})
            continue

        in_args    = [n for n, i in kernel["intent_map"].items() if i == "IN"]
        out_args   = [n for n, i in kernel["intent_map"].items() if i == "OUT"]
        inout_args = [n for n, i in kernel["intent_map"].items() if i == "INOUT"]

        prompt = HumanMessage(content=(
            f"Add OpenACC pragmas to this Fortran subroutine.\n"
            f"Loki analysis:\n"
            f"  loops:         {kernel['loops']}\n"
            f"  INTENT(IN):    {in_args}\n"
            f"  INTENT(OUT):   {out_args}\n"
            f"  INTENT(INOUT): {inout_args}\n"
            f"  array dims:    {kernel['dimensions']}\n\n"
            f"```fortran\n{src}\n```"
        ))
        try:
            resp = llm.invoke([system, prompt])
            annotated = _strip_markdown(resp.content)
            print(f"  {kernel['routine_name']} → OpenACC pragmas inserted")
            updated.append({**kernel, "openacc_code": annotated})
        except Exception as e:
            print(f"  LLM failed for {kernel['routine_name']}: {e}")
            updated.append({**kernel, "openacc_code": src, "error_log": str(e)})

    combined = "\n\n".join(k["openacc_code"] for k in updated)
    _save(_out("fortran_gpu") / "kernel_gpu.f90", combined)

    return {
        "kernel_results": updated,
        "openacc_fortran": combined,
        "executed_agents": list(state.get("executed_agents", [])) + ["openacc"],
    }


# ── Node 4 : Cython Wrapper ──────────────────────────────────────────────────

def cython_wrapper_agent(state: Phase1State) -> dict:
    """LLM : génère un wrapper Cython (.pyx) avec NumPy typed memoryviews."""
    print(f"\n{SEP}")
    print("  [Cython] Generating Python wrapper")
    print(SEP)

    llm = get_llm()

    # Only wrap compute kernels (no I/O)
    eligible = [k for k in state.get("kernel_results", []) if not k["has_io"]]
    if not eligible:
        print("  No eligible routines for Cython wrapping (all have I/O).")
        return {
            "cython_pyx": "",
            "cython_header": "",
            "cython_setup": "",
            "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
        }

    filepath    = state["fortran_filepath"]
    module_name = Path(filepath).stem.lower().replace("-", "_").replace(".", "_")

    routines_summary = [
        {
            "name":       k["routine_name"],
            "intent_map": k["intent_map"],
            "dimensions": k["dimensions"],
        }
        for k in eligible
    ]

    # ── Generate .pyx ────────────────────────────────────────────────
    pyx_system = SystemMessage(content=(
        "You are a Cython expert specializing in Fortran interoperability. "
        "Generate clean, efficient Cython wrappers with correct memory layout."
    ))
    pyx_prompt = HumanMessage(content=(
        f"Generate a Cython wrapper (.pyx) for these Fortran subroutines "
        f"compiled with nvfortran -acc (OpenACC).\n"
        f"Module name: {module_name}\n"
        f"Routines: {routines_summary}\n\n"
        f"Requirements:\n"
        f"  1. cdef extern from 'kernel_c.h' block declaring C signatures\n"
        f"  2. cpdef functions with NumPy typed memoryviews:\n"
        f"       np.float64_t[:] for 1-D arrays\n"
        f"       np.float64_t[:,:] for 2-D arrays\n"
        f"  3. np.asfortranarray() to ensure Fortran column-major layout\n"
        f"  4. import numpy as np and cimport numpy as cnp at the top\n"
        f"  5. # distutils: language = fortran  directive\n"
        f"Return ONLY the .pyx file content."
    ))

    # ── Generate C header ────────────────────────────────────────────
    header_system = SystemMessage(content="You are a C/Fortran interop expert.")
    header_prompt = HumanMessage(content=(
        f"Generate a C header file (kernel_c.h) for these Fortran subroutines "
        f"using iso_c_binding:\n{routines_summary}\n"
        f"Use double* for REAL(8) arrays, int* for INTEGER, void return type.\n"
        f"Add include guards and extern 'C' block.\n"
        f"Return ONLY the header file content."
    ))

    # ── pyproject.toml build config ──────────────────────────────────
    build_content = (
        f"[build-system]\n"
        f'requires = ["setuptools>=68", "Cython>=3.0", "numpy"]\n'
        f'build-backend = "setuptools.backends.legacy:build"\n\n'
        f"[project]\n"
        f'name = "{module_name}_gpu"\n'
        f'version = "0.1.0"\n'
        f'description = "GPU-accelerated Fortran kernel via OpenACC + Cython"\n'
        f'dependencies = ["numpy"]\n\n'
        f"# Build the Cython extension with nvfortran -acc\n"
        f"# Usage: python setup.py build_ext --inplace\n"
    )

    setup_content = (
        f"from setuptools import setup\n"
        f"from Cython.Build import cythonize\n"
        f"from setuptools.extension import Extension\n"
        f"import numpy as np\n\n"
        f'ext = Extension(\n'
        f'    name="{module_name}",\n'
        f'    sources=["cython/{module_name}.pyx", "fortran_gpu/kernel_gpu.f90"],\n'
        f'    include_dirs=["cython", np.get_include()],\n'
        f'    extra_compile_args=["-acc", "-gpu=cc80", "-Minfo=accel"],\n'
        f'    extra_link_args=["-acc", "-gpu=cc80"],\n'
        f'    language="fortran",\n'
        f')\n\n'
        f"setup(name='{module_name}_gpu', ext_modules=cythonize([ext]))\n"
    )

    pyx_code, header_code = "", ""
    try:
        resp_pyx    = llm.invoke([pyx_system, pyx_prompt])
        pyx_code    = _strip_markdown(resp_pyx.content)

        resp_header = llm.invoke([header_system, header_prompt])
        header_code = _strip_markdown(resp_header.content)

        cython_dir = _out("cython")
        _save(cython_dir / f"{module_name}.pyx", pyx_code)
        _save(cython_dir / "kernel_c.h", header_code)
        _save(Path("output") / "pyproject.toml", build_content)
        _save(Path("output") / "setup.py", setup_content)
        print(f"  Generated: {module_name}.pyx, kernel_c.h, pyproject.toml, setup.py")

    except Exception as e:
        print(f"  LLM failed for Cython wrapper: {e}")

    return {
        "cython_pyx":    pyx_code,
        "cython_header": header_code,
        "cython_setup":  build_content,
        "executed_agents": list(state.get("executed_agents", [])) + ["cython_wrapper"],
    }


# ── Node 5 : Validation ──────────────────────────────────────────────────────

def validation_agent(state: Phase1State) -> dict:
    """Compile le Fortran GPU avec nvfortran et construit l'extension Cython."""
    print(f"\n{SEP}")
    print("  [Validation] Compiling Fortran GPU + Cython")
    print(SEP)

    log_lines: List[str] = []
    passed = True

    gpu_fortran = Path("output/fortran_gpu/kernel_gpu.f90").resolve()
    compiler    = _gpu_compiler()

    # ── Step 1 : Compile Fortran GPU ────────────────────────────────
    if not gpu_fortran.exists():
        log_lines.append("SKIP (Fortran): output/fortran_gpu/kernel_gpu.f90 not found")
        passed = False
    elif not compiler:
        log_lines.append("SKIP (Fortran): nvfortran/pgfortran not found — install NVIDIA HPC SDK")
        passed = False
    else:
        so_path = gpu_fortran.parent / "kernel_gpu.so"
        cmd = [compiler, "-acc", "-gpu=cc80", "-shared", "-fPIC",
               "-o", str(so_path), str(gpu_fortran)]
        print(f"  $ {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                log_lines.append(f"OK (Fortran): compiled → {so_path.name}")
                print(f"  Fortran GPU compiled successfully.")
            else:
                log_lines.append(f"FAIL (Fortran): rc={result.returncode}")
                log_lines.append(result.stderr[:800])
                passed = False
                print(f"  Fortran compilation failed (rc={result.returncode}).")
                if result.stderr:
                    print(f"  {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            log_lines.append("FAIL (Fortran): compilation timeout (>120s)")
            passed = False
        except Exception as e:
            log_lines.append(f"FAIL (Fortran): {e}")
            passed = False

    # ── Step 2 : Build Cython extension ─────────────────────────────
    setup_py = Path("output/setup.py").resolve()
    pyx_files = list(Path("output/cython").glob("*.pyx"))

    if not pyx_files:
        log_lines.append("SKIP (Cython): no .pyx file generated")
    elif not setup_py.exists():
        log_lines.append("SKIP (Cython): output/setup.py not found")
    elif not shutil.which("cython") and not shutil.which("cythonize"):
        log_lines.append("SKIP (Cython): Cython not found in PATH")
    else:
        cmd = [sys.executable, str(setup_py), "build_ext", "--inplace"]
        print(f"  $ {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(Path("output").resolve()), timeout=120
            )
            if result.returncode == 0:
                log_lines.append("OK (Cython): extension built")
                print("  Cython extension built successfully.")
            else:
                log_lines.append(f"FAIL (Cython): rc={result.returncode}")
                log_lines.append(result.stderr[:500])
                passed = False
                print(f"  Cython build failed (rc={result.returncode}).")
        except subprocess.TimeoutExpired:
            log_lines.append("FAIL (Cython): build timeout (>120s)")
            passed = False
        except Exception as e:
            log_lines.append(f"FAIL (Cython): {e}")
            passed = False

    # ── Report ───────────────────────────────────────────────────────
    log = "\n".join(log_lines)
    _save(_out("fortran_gpu") / "validation.log", log)
    status_str = "PASSED" if passed else "FAILED (check validation.log)"
    print(f"\n  Validation : {status_str}")
    print(SEP2 + "\n")

    return {
        "validation_passed": passed,
        "validation_log": log,
        "executed_agents": list(state.get("executed_agents", [])) + ["validation"],
    }


# ==========================================
# 4. Construction du Graphe
# ==========================================

workflow_phase1 = StateGraph(Phase1State)

workflow_phase1.add_node("init",           init_phase1)
workflow_phase1.add_node("parser",         parser_phase1)
workflow_phase1.add_node("pure_elemental", pure_elemental_agent)
workflow_phase1.add_node("openacc",        openacc_insert_agent)
workflow_phase1.add_node("cython_wrapper", cython_wrapper_agent)
workflow_phase1.add_node("validation",     validation_agent)

workflow_phase1.set_entry_point("init")
workflow_phase1.add_edge("init",           "parser")
workflow_phase1.add_edge("parser",         "pure_elemental")
workflow_phase1.add_edge("pure_elemental", "openacc")
workflow_phase1.add_edge("openacc",        "cython_wrapper")
workflow_phase1.add_edge("cython_wrapper", "validation")
workflow_phase1.add_edge("validation",     END)

translation_app_phase1 = workflow_phase1.compile()

__all__ = ["translation_app_phase1"]
