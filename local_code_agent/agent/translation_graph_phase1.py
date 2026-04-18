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


def _gfortran_local_check(sources: List[Path]) -> tuple[bool, bool, str]:
    """Vérifie la syntaxe Fortran en deux passes avec gfortran.

    Flavor 1 — CPU (sans pragmas) :
        Retire les directives !$acc et compile avec gfortran -O2 -fsyntax-only.
        Valide la logique Fortran pure indépendamment d'OpenACC.

    Flavor 2 — Avec pragmas :
        Compile directement avec gfortran -fopenacc -fsyntax-only.
        Valide que les directives OpenACC sont syntaxiquement correctes.

    Returns:
        (ok_no_acc, ok_with_acc, log)
    """
    gfc = shutil.which("gfortran")
    if not gfc:
        return False, False, "SKIP (gfortran): not found in PATH — brew install gcc"

    logs: List[str] = [f"gfortran: {gfc}"]
    ok_no_acc   = False
    ok_with_acc = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Flavor 1 : strip !$acc directives ────────────────────────────
        stripped_srcs: List[str] = []
        for src in sources:
            if not src.exists():
                continue
            code = src.read_text(encoding="utf-8")
            stripped = "\n".join(
                line for line in code.splitlines()
                if not re.match(r"^\s*!\$acc", line, re.IGNORECASE)
            )
            dst = tmp / ("stripped_" + src.name)
            dst.write_text(stripped, encoding="utf-8")
            stripped_srcs.append(str(dst))

        cmd1 = [gfc, "-O2", "-fsyntax-only"] + stripped_srcs
        logs.append(f"\nFlavor 1 (CPU, no !$acc): {' '.join(cmd1)}")
        try:
            r = subprocess.run(cmd1, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                ok_no_acc = True
                logs.append("  OK — Fortran logic is syntactically valid.")
            else:
                logs.append(f"  FAIL rc={r.returncode}")
                logs.append("  " + (r.stderr or r.stdout)[:600])
        except Exception as e:
            logs.append(f"  ERROR: {e}")

        # ── Flavor 2 : keep !$acc, use -fopenacc ─────────────────────────
        all_srcs = [str(s) for s in sources if s.exists()]
        cmd2 = [gfc, "-fopenacc", "-fsyntax-only"] + all_srcs
        logs.append(f"\nFlavor 2 (with OpenACC): {' '.join(cmd2)}")
        try:
            r = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                ok_with_acc = True
                logs.append("  OK — OpenACC directives are syntactically valid.")
            else:
                logs.append(f"  FAIL rc={r.returncode}")
                logs.append("  " + (r.stderr or r.stdout)[:600])
        except Exception as e:
            logs.append(f"  ERROR: {e}")

    return ok_no_acc, ok_with_acc, "\n".join(logs)


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


# ── Node 4 : OpenACC Insert ──────────────────────────────────────────────────

def openacc_insert_agent(state: Phase1State) -> dict:
    """LLM : insère les pragmas OpenACC dans les kernels ET la région data du driver.

    Deux cibles :
      1. Chaque subroutine kernel → !$acc parallel loop collapse(2) sur les boucles 2D
      2. Le driver PROGRAM → !$acc data ... région autour du time loop +
         !$acc update host(...) avant chaque bloc I/O périodique
    """
    print(f"\n{SEP}")
    print("  [OpenACC] Inserting OpenACC pragmas (kernels + driver data region)")
    print(SEP)

    llm = get_llm()

    # ── 4a : Kernel subroutines ───────────────────────────────────────────────
    kernel_system = SystemMessage(content=(
        "You are an OpenACC GPU expert for scientific Fortran.\n"
        "Add OpenACC directives to parallelize this subroutine on NVIDIA A100 GPUs.\n"
        "Compiler: nvfortran -acc -gpu=cc80 (Ampere)\n"
        "Guidelines for finite-difference stencil subroutines:\n"
        "  - Add !$acc parallel loop collapse(2) before the outermost 2D loop nest\n"
        "  - Add private(...) clause for scalar temporaries computed inside the loop\n"
        "  - Do NOT add data movement clauses here — handled by the driver !$acc data region\n"
        "  - !$acc end parallel after end of loop nest\n"
        "  - The subroutine does NOT need !$acc routine — it's called from host, not device\n"
        "Return ONLY the annotated Fortran subroutine."
    ))

    updated: List[KernelInfo] = []
    for kernel in state.get("kernel_results", []):
        src = kernel.get("pure_elemental_code") or kernel["fortran_code"]

        if kernel["has_io"]:
            print(f"  Skip {kernel['routine_name']} (has I/O)")
            updated.append({**kernel, "openacc_code": src})
            continue

        # Identify scalar temporaries (not in intent_map → local variables in stencil)
        # These need private() clause
        prompt = HumanMessage(content=(
            f"Add !$acc parallel loop collapse(2) to the 2D loop nest in this subroutine.\n"
            f"INTENT(IN):    {[n for n,i in kernel['intent_map'].items() if i=='IN']}\n"
            f"INTENT(INOUT): {[n for n,i in kernel['intent_map'].items() if i=='INOUT']}\n\n"
            f"```fortran\n{src}\n```"
        ))
        try:
            resp = llm.invoke([kernel_system, prompt])
            annotated = _strip_markdown(resp.content)
            print(f"  {kernel['routine_name']} → !$acc parallel loop inserted")
            updated.append({**kernel, "openacc_code": annotated})
        except Exception as e:
            print(f"  LLM failed for {kernel['routine_name']}: {e}")
            updated.append({**kernel, "openacc_code": src, "error_log": str(e)})

    # ── 4b : Driver data region ───────────────────────────────────────────────
    driver_src = state.get("driver_fortran", "")
    driver_with_acc = ""

    if driver_src:
        driver_system = SystemMessage(content=(
            "You are an OpenACC GPU expert.\n"
            "Add an !$acc data region around the time loop in this Fortran PROGRAM driver.\n"
            "The subroutines inside the loop are already annotated with !$acc parallel loop.\n\n"
            "Guidelines:\n"
            "  - !$acc data copyin(lambda,mu,rho,b_x,b_x_half,b_y,b_y_half,a_x,...) "
            "copy(vx,vy,sigma_xx,sigma_yy,sigma_xy,memory_dvx_dx,...) before the time loop\n"
            "  - INTENT(IN) arrays  → copyin(...)\n"
            "  - INTENT(INOUT) arrays (field + memory arrays) → copy(...)\n"
            "  - Just before each periodic I/O block (if mod(it,IT_DISPLAY)==0): "
            "add !$acc update host(vx,vy) to transfer velocity fields for PRINT/image output\n"
            "  - !$acc end data after end of time loop\n"
            "  - Keep ALL existing code and I/O intact — only add !$acc directives\n"
            "Return ONLY the modified Fortran PROGRAM."
        ))
        driver_prompt = HumanMessage(content=(
            f"Add !$acc data region around the time loop.\n"
            f"Kernel subroutines called inside the loop: {state.get('kernel_names', [])}\n\n"
            f"```fortran\n{driver_src}\n```"
        ))
        try:
            resp = llm.invoke([driver_system, driver_prompt])
            driver_with_acc = _strip_markdown(resp.content)
            _save(_out("fortran_gpu") / "driver_gpu.f90", driver_with_acc)
            print(f"  driver → !$acc data region inserted")
        except Exception as e:
            driver_with_acc = driver_src
            print(f"  LLM failed for driver data region: {e}")
    else:
        print("  No driver.f90 found — skipping driver data region")

    # ── Save annotated MODULE ────────────────────────────────────────────────
    module_combined = "\n\n".join(k["openacc_code"] for k in updated)
    _save(_out("fortran_gpu") / "module_kernels_gpu.f90", module_combined)

    # The "kernel_gpu.f90" target for validation is the full GPU source
    full_gpu = module_combined + ("\n\n" + driver_with_acc if driver_with_acc else "")
    _save(_out("fortran_gpu") / "kernel_gpu.f90", full_gpu)

    return {
        "kernel_results": updated,
        "openacc_fortran": full_gpu,
        "driver_fortran":  driver_with_acc or driver_src,
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


# ── Node 6 : Validation / GPU Compilation ────────────────────────────────────

def _make_makefile(out_dir: Path, module_file: str, driver_file: str,
                   binary_name: str, pyx_name: str) -> str:
    """Génère un Makefile pour la compilation GPU + Cython. Toujours produit."""
    return f"""# GPU Fortran + Cython build — generated by agent-gpu
# Usage : make          (compile Fortran GPU)
#         make cython   (build Cython extension)
#         make clean

FC      = nvfortran
FFLAGS  = -acc -gpu=cc80 -Minfo=accel -fast
TARGET  = {binary_name}
MODULE  = fortran_gpu/{module_file}
DRIVER  = fortran_gpu/{driver_file}

all: $(TARGET)

# Compile MODULE first (produces .mod file), then link with DRIVER
$(TARGET): $(MODULE) $(DRIVER)
\t$(FC) $(FFLAGS) -c $(MODULE) -o module_kernels_gpu.o
\t$(FC) $(FFLAGS) $(DRIVER) module_kernels_gpu.o -o $(TARGET)
\t@echo "\\n=== Compiled OK: $(TARGET) ===\\n"
\t@echo "Run with: ./$(TARGET)"

# Cython wrapper (requires Cython + numpy installed)
cython: cython/setup.py
\tcd cython && python setup.py build_ext --inplace
\t@echo "Cython extension built in cython/"

clean:
\trm -f $(TARGET) *.o *.mod

.PHONY: all cython clean
"""


def _make_compile_script(out_dir: Path, module_file: str, driver_file: str,
                          binary_name: str) -> str:
    """Génère compile_gpu.sh — script autonome pour HPC/Pangea/Azure (A100 ou T4)."""
    return f"""#!/bin/bash
# compile_gpu.sh — Compile Fortran GPU kernels with nvfortran (OpenACC)
# Generated by agent-gpu. Run this on the GPU node (Azure A100/T4, Pangea).
#
# Usage : bash compile_gpu.sh
#         bash compile_gpu.sh --check   (vérifie l'environnement GPU)

set -e

TARGET="{binary_name}"
MODULE="fortran_gpu/{module_file}"
DRIVER="fortran_gpu/{driver_file}"
FC="${{FC:-nvfortran}}"

# ── Détection automatique du GPU (cc80=A100, cc75=T4, cc70=V100) ─────────
detect_gpu_arch() {{
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    case "$GPU" in
        *A100*)  echo "cc80" ;;
        *H100*)  echo "cc90" ;;
        *V100*)  echo "cc70" ;;
        *T4*)    echo "cc75" ;;
        *3090*|*3080*|*3070*) echo "cc86" ;;
        *)       echo "cc70" ;;   # fallback conservateur
    esac
}}

GPU_ARCH=$(detect_gpu_arch)
FFLAGS="-acc -gpu=${{GPU_ARCH}} -Minfo=accel -fast"

# ── Sanity checks ────────────────────────────────────────────────────────
if [ "$1" = "--check" ]; then
    echo "=== Environment check ==="
    which "$FC" && "$FC" --version | head -1 || echo "WARNING: $FC not found"
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU not found")
    echo "GPU  : $GPU_NAME"
    echo "Arch : $GPU_ARCH"
    echo "Files:"
    ls -lh fortran_gpu/*.f90 2>/dev/null || echo "  No .f90 files found"
    exit 0
fi

# ── Compile ──────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "=== Compiling GPU Fortran ==="
echo "  GPU  : $GPU_NAME ($GPU_ARCH)"
echo "  Flags: $FC $FFLAGS"
echo ""

echo "  Step 1: Compile MODULE → module_kernels_gpu.o"
$FC $FFLAGS -c "$MODULE" -o module_kernels_gpu.o

echo "  Step 2: Compile DRIVER + link → $TARGET"
$FC $FFLAGS "$DRIVER" module_kernels_gpu.o -o "$TARGET"

echo ""
echo "=== SUCCESS: ./$TARGET ==="
echo "  Run  : ./$TARGET"
echo "  Watch: nvidia-smi dmon -d 1"
"""


def validation_agent(state: Phase1State) -> dict:
    """Génère Makefile + compile_gpu.sh, puis valide le code généré en 4 niveaux.

    Niveau 0 — gfortran local (toujours, 2 flavors) :
        Flavor 1 : compile sans !$acc (gfortran -O2 -fsyntax-only) → logique Fortran pure
        Flavor 2 : compile avec OpenACC (gfortran -fopenacc -fsyntax-only) → pragmas valides
    Niveau 1 — nvfortran local   : si nvfortran est dans le PATH
    Niveau 2 — compilation SSH   : si AZURE_GPU_HOST est défini dans l'env
    Niveau 3 — Génération seule  : toujours : Makefile + compile_gpu.sh pour lancement manuel
    """
    print(f"\n{SEP}")
    print("  [Validation] gfortran syntax check → GPU compilation")
    print(SEP)

    out_dir     = Path("output").resolve()
    gpu_dir     = out_dir / "fortran_gpu"
    log_lines:  List[str] = []
    compiled    = False

    # Determine filenames — prefer split module+driver, fallback to kernel_gpu.f90
    module_file = "module_kernels_gpu.f90" if (gpu_dir / "module_kernels_gpu.f90").exists() else "kernel_gpu.f90"
    driver_file = "driver_gpu.f90"         if (gpu_dir / "driver_gpu.f90").exists()         else ""
    filepath    = state.get("fortran_filepath", "kernel")
    binary_name = Path(filepath).stem.lower().replace("-", "_") + "_gpu"

    # ── Always generate build artifacts ──────────────────────────────
    makefile_content = _make_makefile(out_dir, module_file, driver_file or module_file,
                                      binary_name, "")
    compile_sh       = _make_compile_script(out_dir, module_file, driver_file or module_file,
                                            binary_name)
    _save(out_dir / "Makefile",         makefile_content)
    _save(out_dir / "compile_gpu.sh",   compile_sh)
    os.chmod(out_dir / "compile_gpu.sh", 0o755)
    log_lines.append("OK: Makefile and compile_gpu.sh generated in output/")
    print("  Generated: output/Makefile  +  output/compile_gpu.sh")

    # ── Level 0 : gfortran local syntax check (2 flavors) ────────────
    print(f"\n  [Level 0] gfortran local syntax check ...")
    sources_to_check: List[Path] = []
    if (gpu_dir / module_file).exists():
        sources_to_check.append(gpu_dir / module_file)
    if driver_file and (gpu_dir / driver_file).exists():
        sources_to_check.append(gpu_dir / driver_file)

    if sources_to_check:
        ok_no_acc, ok_with_acc, gfc_log = _gfortran_local_check(sources_to_check)
        log_lines.append("\n=== gfortran local check ===")
        log_lines.append(gfc_log)
        flavor1_status = "OK" if ok_no_acc   else "FAIL"
        flavor2_status = "OK" if ok_with_acc else "FAIL"
        print(f"  Flavor 1 (CPU, no !$acc pragmas) : {flavor1_status}")
        print(f"  Flavor 2 (gfortran -fopenacc)    : {flavor2_status}")
        if not ok_no_acc:
            print("  !! Fortran syntax errors — fix before GPU compilation")
        elif not ok_with_acc:
            print("  !! OpenACC directive errors (Fortran logic OK)")
    else:
        log_lines.append("SKIP (gfortran): source files not yet written")
        print("  Level 0 skipped: no source files found")

    # ── Build command list ────────────────────────────────────────────
    def _compile_cmds(fc: str) -> List[List[str]]:
        fflags = ["-acc", "-gpu=cc80", "-Minfo=accel", "-fast"]
        mod_src = str(gpu_dir / module_file)
        if driver_file and (gpu_dir / driver_file).exists():
            drv_src = str(gpu_dir / driver_file)
            return [
                [fc] + fflags + ["-c", mod_src, "-o", str(out_dir / "module_kernels_gpu.o")],
                [fc] + fflags + [drv_src, str(out_dir / "module_kernels_gpu.o"),
                                 "-o", str(out_dir / binary_name)],
            ]
        else:
            return [[fc] + fflags + [mod_src, "-o", str(out_dir / binary_name)]]

    def _run_cmds(cmds: List[List[str]], label: str) -> bool:
        for cmd in cmds:
            print(f"  $ {' '.join(cmd)}")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                   cwd=str(out_dir))
                if r.returncode != 0:
                    log_lines.append(f"FAIL ({label}): {cmd[0]} rc={r.returncode}")
                    log_lines.append(r.stderr[:600])
                    print(f"  FAIL rc={r.returncode}: {r.stderr[:200]}")
                    return False
                if r.stdout:
                    log_lines.append(r.stdout[:300])
            except subprocess.TimeoutExpired:
                log_lines.append(f"FAIL ({label}): timeout >180s")
                return False
            except Exception as e:
                log_lines.append(f"FAIL ({label}): {e}")
                return False
        log_lines.append(f"OK ({label}): compiled → {binary_name}")
        return True

    # ── Level 1 : Local nvfortran ─────────────────────────────────────
    local_fc = _gpu_compiler()
    if local_fc and (gpu_dir / module_file).exists():
        print(f"\n  Attempting local compilation with {local_fc} ...")
        if _run_cmds(_compile_cmds(local_fc), "local"):
            compiled = True
            print(f"  LOCAL compilation OK → output/{binary_name}")
        else:
            print("  Local compilation failed — see validation.log")
    else:
        reason = "nvfortran not found" if not local_fc else f"{module_file} not found"
        log_lines.append(f"SKIP (local): {reason}")
        print(f"  Local compile skipped: {reason}")

    # ── Level 2 : SSH remote (Azure A100 / Pangea) ────────────────────
    gpu_host = os.getenv("AZURE_GPU_HOST", "")
    gpu_user = os.getenv("AZURE_GPU_USER", "azureuser")
    gpu_key  = os.getenv("AZURE_GPU_KEY",  "")   # path to SSH private key

    if not compiled and gpu_host:
        print(f"\n  Attempting remote compilation on {gpu_user}@{gpu_host} ...")
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]
        if gpu_key:
            ssh_opts += ["-i", gpu_key]

        remote_dir = f"~/gpu_agent_build/{binary_name}"
        try:
            # Copy output/ to remote
            scp_cmd = ["scp"] + ssh_opts + ["-r", str(out_dir),
                       f"{gpu_user}@{gpu_host}:{remote_dir}"]
            print(f"  scp output/ → {gpu_host}:{remote_dir}")
            r = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError(f"scp failed: {r.stderr[:200]}")

            # Run make on remote
            make_cmd = ["ssh"] + ssh_opts + [f"{gpu_user}@{gpu_host}",
                        f"cd {remote_dir}/output && bash compile_gpu.sh"]
            print(f"  ssh → bash compile_gpu.sh")
            r = subprocess.run(make_cmd, capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                compiled = True
                log_lines.append(f"OK (SSH {gpu_host}): compiled → {binary_name}")
                print(f"  REMOTE compilation OK on {gpu_host}")
                print(f"  Binary: {remote_dir}/output/{binary_name}")
            else:
                log_lines.append(f"FAIL (SSH): rc={r.returncode}\n{r.stderr[:400]}")
                print(f"  Remote compilation failed: {r.stderr[:200]}")
        except Exception as e:
            log_lines.append(f"FAIL (SSH): {e}")
            print(f"  SSH error: {e}")
    elif not compiled and not gpu_host:
        log_lines.append(
            "SKIP (SSH): set AZURE_GPU_HOST=<ip> (+ AZURE_GPU_USER, AZURE_GPU_KEY) "
            "to enable remote compilation"
        )
        print("  SSH skipped: AZURE_GPU_HOST not set")

    # ── Level 3 : Manual instructions ────────────────────────────────
    if not compiled:
        log_lines.append("")
        log_lines.append("=== Manual compilation (copy output/ to GPU node) ===")
        log_lines.append(f"  scp -r output/ azureuser@<GPU_IP>:~/seismic_gpu/")
        log_lines.append(f"  ssh azureuser@<GPU_IP>")
        log_lines.append(f"  cd seismic_gpu && bash compile_gpu.sh")
        log_lines.append(f"  ./{binary_name}")
        print("\n  Manual steps written to validation.log")

    # ── Report ────────────────────────────────────────────────────────
    log = "\n".join(log_lines)
    _save(gpu_dir / "validation.log", log)

    print(f"\n  Files generated in output/:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file() and not f.name.endswith(".pyc"):
            size = f.stat().st_size
            rel  = f.relative_to(out_dir)
            ok   = "OK" if size > 50 else "??"
            print(f"    [{ok}] {str(rel):<45} {size:>7} bytes")

    status = "COMPILED" if compiled else "ARTIFACTS_READY (run compile_gpu.sh on GPU node)"
    print(f"\n  Status : {status}")
    print(SEP2 + "\n")

    return {
        "validation_passed": compiled,
        "validation_log":    log,
        "executed_agents":   list(state.get("executed_agents", [])) + ["validation"],
    }


# ==========================================
# 4. Construction du Graphe
# ==========================================

workflow_phase1 = StateGraph(Phase1State)

workflow_phase1.add_node("init",           init_phase1)
workflow_phase1.add_node("parser",         parser_phase1)
workflow_phase1.add_node("extractor",      extractor_agent)
workflow_phase1.add_node("pure_elemental", pure_elemental_agent)
workflow_phase1.add_node("openacc",        openacc_insert_agent)
workflow_phase1.add_node("cython_wrapper", cython_wrapper_agent)
workflow_phase1.add_node("validation",     validation_agent)

workflow_phase1.set_entry_point("init")
workflow_phase1.add_edge("init",           "parser")
workflow_phase1.add_edge("parser",         "extractor")
workflow_phase1.add_edge("extractor",      "pure_elemental")
workflow_phase1.add_edge("pure_elemental", "openacc")
workflow_phase1.add_edge("openacc",        "cython_wrapper")
workflow_phase1.add_edge("cython_wrapper", "validation")
workflow_phase1.add_edge("validation",     END)

translation_app_phase1 = workflow_phase1.compile()

__all__ = ["translation_app_phase1"]
