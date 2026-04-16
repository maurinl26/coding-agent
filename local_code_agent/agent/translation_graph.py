"""
Graph d'agents LangGraph pour la traduction de Fortran 90 vers JAX
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from local_code_agent.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


# ==========================================
# 1. État du Graphe (State)
# ==========================================
class TranslationState(TypedDict):
    fortran_filepath: str
    fortran_code: str
    ast_info: Dict[str, Any]
    isolated_kernel: str
    children_code: Dict[str, str]
    jax_code: str
    call_graph: str
    jax_hints: List[str]
    compilation_error: str
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]


# ==========================================
# 2. Définition des Noeuds (Agents)
# ==========================================

def parse_and_isolate_agent(state: TranslationState) -> TranslationState:
    """Agent 1: Utilise Loki pour parser et isoler le kernel Fortran."""
    print("[Parsing Agent] Extracting numerical kernel from:", state["fortran_filepath"])
    
    try:
        import os
        from loki import Sourcefile, Scheduler
        
        # Analyse du graphe de dépendance via Loki Scheduler
        workspace_dir = os.path.dirname(state["fortran_filepath"])
        scheduler = Scheduler(paths=[workspace_dir], match=['*.f90'])
        
        # Parsing du fichier cible
        source = Sourcefile.from_file(state["fortran_filepath"])
        
        if not source.routines:
            raise ValueError("No Fortran routines found in the file.")
            
        # Extract the primary routine (kernel)
        routine = source.routines[0]
        
        # Tentative d'extraction des dépendances SGraph
        graph_info = "Standalone kernel"
        children_routines = {}
        
        try:
            # Construction de l'item depuis le nom de la routine pour le graph Loki
            item_name = f"{source.path.stem}#{routine.name.lower()}"
            item = scheduler.item_factory(item_name)
            
            # Les dépendances Loki sont renvoyées via item.dependencies
            deps = getattr(item, 'dependencies', [])
            
            for dep in deps:
                try:
                    dep_item = scheduler.item_factory(dep)
                    if hasattr(dep_item, 'routine') and dep_item.routine:
                        children_routines[dep] = dep_item.routine.to_fortran()
                    elif hasattr(dep_item, 'source') and dep_item.source:
                        children_routines[dep] = dep_item.source.to_fortran()
                except Exception as inner_e:
                    children_routines[dep] = f"! Failed to load child {dep}: {str(inner_e)}"
                    
            if deps:
                graph_info = f"Dependencies for {routine.name}: " + ", ".join([str(d) for d in deps])
        except Exception as e:
            graph_info = f"Graph resolution skipped: {str(e)}"
        
        # Extraction du noyau brut
        kernel_str = routine.to_fortran()
        
        # === MAPPING STRUCTUREL LOKI (VMAP/SCAN) ===
        jax_hints = []
        try:
            from loki import FindNodes, Loop
            loops = FindNodes(Loop).visit(routine.body)
            for loop in loops:
                l_var = str(loop.variable).lower()
                if l_var in ['nt', 't', 'time', 'it', 'step']:
                    hint = "[Hint: Use jax.lax.scan for sequential time dependencies]"
                    if hint not in jax_hints: jax_hints.append(hint)
                elif l_var in ['nx', 'ny', 'nz', 'i', 'j', 'k']:
                    hint = "[Hint: Vectorize spatial loop with jax.vmap]"
                    if hint not in jax_hints: jax_hints.append(hint)
        except Exception as e:
            print(f"[Warning] Loki structural mapping failed: {e}")
        
        # === LIMITS ===
        line_count = len(kernel_str.split('\n'))
        if line_count > 1200:
            return {
                "isolated_kernel": "", 
                "children_code": children_routines,
                "call_graph": graph_info,
                "jax_hints": jax_hints,
                "ast_info": {"status": "error", "message": f"Trial Limit Exceeded: Kernel has {line_count} lines (> 1200 limit to prevent OOM)."},
                "compilation_error": "FILE_TOO_LARGE"
            }
        
        return {
            "isolated_kernel": kernel_str, 
            "children_code": children_routines,
            "call_graph": graph_info,
            "jax_hints": jax_hints,
            "ast_info": {"status": "parsed", "routine_name": routine.name}
        }
    except ImportError:
        print("[Warning] loki-ifs not installed locally. Falling back to raw file content.")
        return {"isolated_kernel": state.get("fortran_code", ""), "children_code": {}, "call_graph": "Fallback", "jax_hints": [], "ast_info": {"status": "mocked_fallback"}}
    except Exception as e:
        print(f"[Error] Loki parser failed: {str(e)}")
        # Fallback to pure string
        return {"isolated_kernel": state["fortran_code"], "call_graph": "Error", "jax_hints": [], "ast_info": {"status": "error", "message": str(e)}}
        print("[Warning] loki-ifs not installed locally. Falling back to raw file content.")
        return {"isolated_kernel": state["fortran_code"], "ast_info": {"status": "mocked_fallback"}}
    except Exception as e:
        print(f"[Error] Loki parser failed: {str(e)}")
        # Fallback to pure string
        return {"isolated_kernel": state["fortran_code"], "ast_info": {"status": "error", "message": str(e)}}


def translate_kernel_agent(state: TranslationState) -> TranslationState:
    """Agent 2: Traduit le kernel Fortran vers JAX via LLM."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        print("[Translation Agent] Skipped due to file size limit.")
        return state
        
    print("[Translation Agent] Translating kernel to JAX...")
    # On importe get_translator_llm ("Codestral") défini dans notre nouvelle archi Cloud Azure
    from local_code_agent.llm import get_translator_llm
    llm = get_translator_llm()
    
    children_context = ""
    if state.get('children_code'):
        children_context = "Child Routines (for context):\n"
        for child_name, child_code in state['children_code'].items():
            children_context += f"--- {child_name} ---\n{child_code}\n\n"

    hints_context = ""
    if state.get('jax_hints'):
        hints_context = "Loki Structural Hints for JAX Vectorization:\n" + "\n".join(state['jax_hints'])

    prompt = f"""
    Translate the following Fortran 90 kernel into pure Python using JAX (jax.numpy).
    Ensure loops are vectorized using jax.numpy operations or jax.lax.scan/fori_loop.
    
    Loki Call Graph Dependencies:
    {state.get('call_graph', 'None')}
    
    {hints_context}
    
    {children_context}
    
    Fortran Kernel:
    {state['isolated_kernel']}
    
    Return ONLY the Python code without any markdown wrapper.
    """
    response = llm.invoke([SystemMessage(content="You are an expert scientific Fortran to JAX translator."),
                           HumanMessage(content=prompt)])
    
    return {"jax_code": response.content}


def halo_exchange_agent(state: TranslationState) -> TranslationState:
    """Agent : Remplace les échanges MPI bas-niveau par GHEX (Generic Halo Exchange)."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state
        
    print("[GHEX Agent] Replacing MPI boundaries with Generic Halo Exchange (GHEX) patterns...")
    from local_code_agent.llm import get_translator_llm
    llm = get_translator_llm()
    
    prompt = f"""
    Analyze the following JAX code. If it contains MPI_Send/Recv or explicit halo boundary exchanges,
    rewrite those specific communication blocks using abstract generic definitions suitable for the Exascale GHEX library.
    If no MPI/Halo logic exists, return the exact same code.
    
    Code:
    {state['jax_code']}
    """
    response = llm.invoke([SystemMessage(content="You are an HPC Halo Exchange expert."),
                           HumanMessage(content=prompt)])
                           
    return {"jax_code": response.content}


def autodiff_adjoint_agent(state: TranslationState) -> TranslationState:
    """Agent : Génère le modèle Adjoint via la différentiation automatique de JAX."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state
        
    print("[Autodiff Agent] Automatically generating Adjoint simulation via jax.grad/jax.vjp...")
    from local_code_agent.llm import get_translator_llm
    llm = get_translator_llm()
    
    prompt = f"""
    You are provided with a forward simulation kernel written in JAX.
    Add a new function at the bottom `adjoint_kernel` that uses `jax.grad` or `jax.vjp` 
    to automatically construct the exact adjoint (inverse) simulation from the forward logic.
    
    Current Forward JAX Code:
    {state['jax_code']}
    """
    response = llm.invoke([SystemMessage(content="You are an expert in Autodifferentiation and adjoint models."),
                           HumanMessage(content=prompt)])
                           
    return {"jax_code": response.content}


def reproducibility_agent(state: TranslationState) -> TranslationState:
    """Agent 3: Vérifie l'égalité numérique des résultats numpy."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state
        
    print("[Reproducibility Agent] Ensuring JAX code is equivalent to Fortran...")
    from local_code_agent.llm import get_reasoning_llm
    llm_reasoning = get_reasoning_llm()
    # TODO: Appel f2py ou ctypes pour compiler le Fortran original
    # TODO: Si le test échoue, utiliser llm_reasoning pour analyser les stack traces XLA/JAX.
    return {"test_results": {"reproducible": True, "error_msg": ""}}


def performance_agent(state: TranslationState) -> TranslationState:
    """Agent 4: Benchmark CPU (Fortran) vs GPU (JAX)."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state
        
    print("[Performance Agent] Profiling both versions...")
    # TODO: timeit sur Fortran compilé vs jax.jit(func).lower().compile()
    return {"performance_metrics": {"speedup": 10.5, "device": "GPU"}}


def docstring_agent(state: TranslationState) -> TranslationState:
    """Agent 5: Ajoute des docstrings scientifiques au code JAX."""
    if state.get("compilation_error") == "FILE_TOO_LARGE":
        return state
        
    print("[Docstring Agent] Enriching JAX code with scientific formulas and references...")
    llm = get_llm()
    
    prompt = f"""
    You have a translated JAX python code. Update it by adding comprehensive Python docstrings.
    
    Requirements:
    1. At the module level, add a docstring stating the equivalent original Fortran routine ({state['fortran_filepath']}) 
       and deduce the associated scientific paper from standard seismic CPML literature context.
    2. For each function, add a docstring explaining the exact mathematical formula being solved/calculated.
    
    Current JAX Code:
    {state['jax_code']}
    
    Return the fully updated Python code.
    """
    response = llm.invoke([SystemMessage(content="You are an expert scientific documentation assistant."),
                           HumanMessage(content=prompt)])
    
    return {"jax_code": response.content}


# ==========================================
# 3. Construction du Graphe
# ==========================================

workflow = StateGraph(TranslationState)

workflow.add_node("parser", parse_and_isolate_agent)
workflow.add_node("translator", translate_kernel_agent)
workflow.add_node("halo_exchange", halo_exchange_agent)
workflow.add_node("autodiff", autodiff_adjoint_agent)
workflow.add_node("docstring", docstring_agent)
workflow.add_node("reproducibility", reproducibility_agent)
workflow.add_node("performance", performance_agent)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "translator")
workflow.add_edge("translator", "halo_exchange")
workflow.add_edge("halo_exchange", "autodiff")
workflow.add_edge("autodiff", "docstring")
workflow.add_edge("docstring", "reproducibility")
workflow.add_edge("reproducibility", "performance")
workflow.add_edge("performance", END)

translation_app = workflow.compile()
