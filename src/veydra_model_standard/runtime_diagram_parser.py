"""
In-memory AST parser for runtime flow-diagram and feedback-loop generation.

This module is intentionally filesystem-free so it can run in worker pipelines and
inside Pyodide with the same core logic.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Set, Tuple

from .feedback_loops import create_cld_data_structure, find_feedback_loops
from .veydra_ast_parser import VeydraModelASTParser

CATEGORY_MAP = {
    "stock": "stock",
    "flow": "flow",
    "rate": "flow",
    "parameter": "parameter",
    "auxiliary": "auxiliary",
    "variable": "auxiliary",
    "constant": "constant",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\\.]*")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def normalize_category(raw: Any) -> str:
    key = str(raw or "").strip().lower()
    return CATEGORY_MAP.get(key, "auxiliary")


def normalize_namespace(file_path: str) -> str:
    parts = [p for p in str(file_path).split("/") if p]
    if len(parts) >= 3 and parts[1] == "src":
        file_name = parts[-1]
        if file_name in ("model.py", "__init__.py"):
            return "main"
        return file_name.replace(".py", "") or "main"
    return "main"


def normalize_token(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    if value.startswith("self."):
        value = value[5:]
    if value.startswith("context."):
        value = value[8:]
    if value.startswith("model."):
        value = value[6:]
    if value.startswith("simulation."):
        return ""
    return value


def boundary_flow_base_name(flow_name: str) -> str:
    tail = str(flow_name or "").split(".")[-1]
    return tail.replace("_flow", "").replace("_rate", "")


def _get_call_name(call_node: ast.Call) -> str:
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return func.attr
    return ""


def alias_candidates(variable_name: str) -> Set[str]:
    value = str(variable_name or "").strip()
    if not value:
        return set()

    candidates = {value}
    tail = value.split(".")[-1]
    candidates.add(tail)

    if tail.startswith("s_"):
        stock_tail = tail[2:]
        candidates.add(stock_tail)
        candidates.add(f"d_{stock_tail}_dt")

    if tail.endswith("_dt"):
        candidates.add(tail[:-3])

    return {candidate for candidate in candidates if candidate}


def build_alias_index(variables: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
    alias_index: Dict[str, Set[str]] = {}
    for var_name in variables.keys():
        for candidate in alias_candidates(var_name):
            alias_index.setdefault(candidate, set()).add(var_name)
    return alias_index


def resolve_dependency_targets(
    dependency_name: str,
    file_dependencies: Dict[str, List[str]],
    alias_index: Dict[str, Set[str]],
    visited: Set[str] | None = None,
    exclude: Set[str] | None = None,
) -> Set[str]:
    """The published variables ``dependency_name`` refers to, nearest-first.

    Recursion exists to see through *local* temporaries — ``water_level`` ->
    ``bathtub.s_water_level``, ``actual_drain_flow`` -> its inputs. It stops as
    soon as a name resolves to a published variable: walking past one emits
    transitive shortcut edges that skip the mediating node (e.g.
    ``s_predator_population -> s_prey_population`` jumping over
    ``prey_loss_rate_dt``), which surface as duplicate feedback loops beside the
    real mediated one.

    ``exclude`` names cannot terminate the walk, so resolving a target's own
    dependencies keeps expanding through the target's identity aliases instead
    of stopping on itself.
    """
    if not dependency_name:
        return set()

    if visited is None:
        visited = set()

    if dependency_name in visited:
        return set()

    visited.add(dependency_name)

    direct = {name for name in alias_index.get(dependency_name, set()) if not exclude or name not in exclude}
    if direct:
        return direct

    resolved: Set[str] = set()
    for child_dependency in file_dependencies.get(dependency_name, []):
        resolved.update(
            resolve_dependency_targets(child_dependency, file_dependencies, alias_index, visited, exclude)
        )
    return resolved


def resolve_flow_variable(flow_name: str, stock_name: str, stock_info: Dict[str, Any], variables: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    raw = str(flow_name or "").strip()
    if not raw:
        return None

    stock_namespace = stock_info.get("namespace") or "main"
    stock_domain = stock_name.rsplit(".", 1)[0] if "." in stock_name else ""
    raw_tail = raw.split(".")[-1]

    candidates = [raw]
    if stock_domain and "." not in raw:
        candidates.append(f"{stock_domain}.{raw}")
    if "." not in raw:
        candidates.append(f"{stock_namespace}.{raw}")
    candidates.append(raw_tail)
    if stock_domain:
        candidates.append(f"{stock_domain}.{raw_tail}")
    candidates.append(f"{stock_namespace}.{raw_tail}")

    for candidate in candidates:
        flow = variables.get(candidate)
        if flow and flow.get("type") == "flow":
            return flow

    for var_name, info in variables.items():
        if info.get("type") != "flow":
            continue
        if var_name.split(".")[-1] == raw_tail:
            return info

    return None


def _resolves_to_flow(
    dependency_name: str,
    file_dependencies: Dict[str, List[str]],
    alias_index: Dict[str, Set[str]],
    variables: Dict[str, Dict[str, Any]],
    exclude: Set[str] | None = None,
) -> bool:
    for source_name in resolve_dependency_targets(dependency_name, file_dependencies, alias_index, None, exclude):
        if variables.get(source_name, {}).get("type") == "flow":
            return True
    return False


def _extract_expression_dependencies(expr_node: ast.AST) -> List[str]:
    dependencies: List[str] = []

    for node in ast.walk(expr_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            var_name = node.id
            if var_name not in ["self", "sim_context", "True", "False", "None", "min", "max", "abs", "sum", "len", "range", "float", "int"]:
                if var_name not in dependencies:
                    dependencies.append(var_name)
        elif isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            include_first_string_arg = call_name in {
                "sim_context.get_param",
                "sim_context.get_stock",
                "sim_context.get_flow",
                "upstream_flows.get",
                "disruptor_flows.get",
            }

            for arg in node.args:
                if isinstance(arg, ast.Name) and isinstance(arg.ctx, ast.Load):
                    var_name = arg.id
                    if var_name not in ["self", "sim_context"] and var_name not in dependencies:
                        dependencies.append(var_name)

            for arg_index, arg in enumerate(node.args):
                if not include_first_string_arg or arg_index != 0:
                    continue
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value not in dependencies:
                        dependencies.append(arg.value)

            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and isinstance(kw.value.ctx, ast.Load):
                    var_name = kw.value.id
                    if var_name not in ["self", "sim_context"] and var_name not in dependencies:
                        dependencies.append(var_name)

            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sim_context"
                and node.func.attr in ["get_param", "get_stock", "get_flow"]
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                param_name = node.args[0].value
                if isinstance(param_name, str) and param_name not in dependencies:
                    dependencies.append(param_name)

    return dependencies


def _collect_local_variable_dependencies(tree: ast.AST) -> Dict[str, List[str]]:
    """Dependencies of every ``name = expr`` assignment, unioned across re-assignments.

    VeydraModelASTParser keeps one dependency list per name and lets a later
    assignment overwrite an earlier one, so a clamp such as::

        actual_drain_flow = drain_flow_rate + drain_sensitivity * water_level
        actual_drain_flow = max(0.0, actual_drain_flow)

    leaves ``actual_drain_flow`` depending only on itself and the whole causal
    chain back to the stock is lost. Unioning every assignment — and dropping
    self-references, which add no information — keeps the chain intact.
    """
    dependencies: Dict[str, List[str]] = {}

    def _record(name: str, deps: List[str]) -> None:
        bucket = dependencies.setdefault(name, [])
        for dep in deps:
            if dep != name and dep not in bucket:
                bucket.append(dep)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: List[ast.AST] = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue

        deps = _extract_expression_dependencies(node.value)
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            _record(target.id, deps)

    # Derivatives/flows mapping-dict entries ({'ns.stock': d_stock_dt, ...}) are
    # dependency assignments too. Scan every dict literal — the mapping may be
    # assigned to a named local OR returned inline (`return {...}, {...}`);
    # requiring the named form silently dropped stock dependencies for
    # inline-return models. Entries whose value is a dict/constant are VARIABLES
    # metadata specs, not expressions, and are skipped.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values):
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                continue
            if isinstance(value_node, (ast.Dict, ast.Constant)):
                continue
            # Skip self-aliases ({'ns.price_change_flow': price_change_flow}) —
            # recording full-name -> short-name would splice a pass-through hop
            # into alias resolution and break flow-terminal detection.
            key_tail = key_node.value.split('.')[-1]
            if isinstance(value_node, ast.Name) and value_node.id == key_tail:
                continue
            deps = [d for d in _extract_expression_dependencies(value_node) if d != key_tail]
            if deps:
                _record(key_node.value, deps)

    return dependencies


def _merge_variable_dependencies(
    parsed: Dict[str, List[str]],
    local: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Union the AST parser's dependency map with the locally recovered one."""
    merged: Dict[str, List[str]] = {}
    for name in set(parsed) | set(local):
        bucket: List[str] = []
        for dep in list(parsed.get(name) or []) + list(local.get(name) or []):
            if dep and dep != name and dep not in bucket:
                bucket.append(dep)
        merged[name] = bucket
    return merged


def _expression_identity_name(expression: Any) -> str:
    """The bare name a variable is assigned from, e.g. ``drain_dt = actual_drain_flow``.

    Returns "" for anything with real arithmetic in it.
    """
    raw = str(expression or "").strip()
    rhs = raw.split("=", 1)[1].strip() if "=" in raw else raw
    return rhs if IDENTIFIER_PATTERN.match(rhs) else ""


def _flow_identity_aliases(
    flow_expressions: Dict[str, Any],
    variables: Dict[str, Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """Locals a published flow is *defined as* -> that flow.

    Stock derivatives often reference the intermediate local
    (``d_water_level_dt = tap_flow_rate - actual_drain_flow``) rather than the
    published flow id, while the flow is published separately
    (``drain_dt = actual_drain_flow``). Without mapping the local back to the
    flow, the flow -> stock accumulation link, and the loop it closes, is invisible.
    """
    aliases: Dict[str, Set[str]] = {}
    for flow_name, expression in (flow_expressions or {}).items():
        if variables.get(flow_name, {}).get("type") != "flow":
            continue
        local = _expression_identity_name(expression)
        if not local or local in alias_candidates(flow_name):
            continue  # already reachable under the flow's own name
        aliases.setdefault(local, set()).add(flow_name)
    return aliases


def _short_var_segment(name: str) -> str:
    """Last identifier segment of a variable name, stripping namespace/prefix.

    Signed dependencies mix local names (``patients``) with namespaced keys
    (``core::patients`` / ``core.patients``); comparing on the trailing segment
    lets computational-edge polarity match regardless of which form each side uses.
    """
    return str(name or "").split("::")[-1].split(".")[-1]


def _extract_signed_dependencies(expr_node: ast.AST, sign: int = 1, out: List[Tuple[str, int]] | None = None) -> List[Tuple[str, int]]:
    if out is None:
        out = []

    if isinstance(expr_node, ast.BinOp):
        if isinstance(expr_node.op, ast.Add):
            _extract_signed_dependencies(expr_node.left, sign, out)
            _extract_signed_dependencies(expr_node.right, sign, out)
            return out
        if isinstance(expr_node.op, ast.Sub):
            _extract_signed_dependencies(expr_node.left, sign, out)
            _extract_signed_dependencies(expr_node.right, -sign, out)
            return out
        if isinstance(expr_node.op, ast.Div):
            # Numerator keeps sign; a variable in the denominator has the opposite
            # monotonic effect on the result, so flip its sign.
            _extract_signed_dependencies(expr_node.left, sign, out)
            _extract_signed_dependencies(expr_node.right, -sign, out)
            return out
        if isinstance(expr_node.op, ast.Mult):
            # Factors are assumed positive (the standard SD reading), so each
            # factor's internal signs carry through the product unchanged.
            # Without this, `rate * (1 - stock / capacity)` falls through to the
            # generic path below, which flattens the whole tree to "+" and reads
            # a logistic brake as reinforcing.
            _extract_signed_dependencies(expr_node.left, sign, out)
            _extract_signed_dependencies(expr_node.right, sign, out)
            return out

    if isinstance(expr_node, ast.UnaryOp):
        if isinstance(expr_node.op, ast.USub):
            _extract_signed_dependencies(expr_node.operand, -sign, out)
            return out
        if isinstance(expr_node.op, ast.UAdd):
            _extract_signed_dependencies(expr_node.operand, sign, out)
            return out

    if isinstance(expr_node, ast.Call):
        call_name = _get_call_name(expr_node)
        include_first_string_arg = call_name in {
            "sim_context.get_param",
            "sim_context.get_stock",
            "sim_context.get_flow",
            "upstream_flows.get",
            "disruptor_flows.get",
        }

        for arg_index, arg in enumerate(expr_node.args):
            if include_first_string_arg and arg_index == 0:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.append((arg.value, sign))
                continue
            _extract_signed_dependencies(arg, sign, out)

        for kw in expr_node.keywords:
            _extract_signed_dependencies(kw.value, sign, out)

        return out

    for dep in _extract_expression_dependencies(expr_node):
        out.append((dep, sign))
    return out


def _extract_target_signed_dependencies_from_tree(tree: ast.AST, target_ids: Set[str]) -> Dict[str, List[Tuple[str, int]]]:
    target_dependencies: Dict[str, List[Tuple[str, int]]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                target_id = key_node.value
                if target_id not in target_ids:
                    continue
                target_dependencies.setdefault(target_id, [])
                for dep, dep_sign in _extract_signed_dependencies(value_node):
                    pair = (dep, dep_sign)
                    if pair not in target_dependencies[target_id]:
                        target_dependencies[target_id].append(pair)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if not (isinstance(target.value, ast.Name) and target.value.id in ["flows", "derivatives", "auxiliaries"]):
                    continue
                slice_node = target.slice
                if isinstance(slice_node, ast.Index):
                    slice_node = slice_node.value
                if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                    target_id = slice_node.value
                    if target_id in target_ids:
                        target_dependencies.setdefault(target_id, [])
                        for dep, dep_sign in _extract_signed_dependencies(node.value):
                            pair = (dep, dep_sign)
                            if pair not in target_dependencies[target_id]:
                                target_dependencies[target_id].append(pair)

    return target_dependencies


def _resolve_computational_signs_from_tree(
    tree: ast.AST,
    published: Set[str],
    flow_aliases: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, int]]:
    """Signed dependencies for each target, following local assignments.

    Generated submodels compute intermediate locals (``treatment_flow = capacity -
    burnout``) then return ``{'core.treatment_flow': treatment_flow}``. Reading only
    the return dict loses the sign, so we: (1) collect every ``name = expr`` local
    assignment's signed deps, (2) for each published target, expand its defining
    expression one forced level (the dict value's local), then resolve remaining
    purely-local temporaries transitively — stopping at any published variable so
    real edge endpoints (e.g. burnout) keep their direct sign. Keyed by short name.

    ``flow_aliases`` maps a local that *is* a published flow (``drain_dt =
    actual_drain_flow``) to that flow's short name, so a derivative term spelled
    as the local carries its sign onto the flow instead of being expanded past it
    into the flow's own inputs.
    """
    flow_aliases = flow_aliases or {}
    local_assign: Dict[str, List[Tuple[str, int]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # Union across re-assignments, dropping self-references: a clamp like
            # `x = max(0.0, x)` must not erase `x = a - b`'s signed dependencies.
            name = node.targets[0].id
            bucket = local_assign.setdefault(name, [])
            for pair in _extract_signed_dependencies(node.value):
                if _short_var_segment(pair[0]) == name or pair in bucket:
                    continue
                bucket.append(pair)

    def _resolve(deps: List[Tuple[str, int]], seen: frozenset, target_short: str) -> List[Tuple[str, int]]:
        resolved: List[Tuple[str, int]] = []
        for dep, sign in deps:
            short = _short_var_segment(dep)
            alias = flow_aliases.get(short)
            if alias and alias != short and alias != target_short:
                resolved.append((alias, sign))  # the local *is* this published flow
            elif short not in published and short in local_assign and short not in seen:
                for d2, s2 in _resolve(local_assign[short], seen | {short}, target_short):
                    resolved.append((d2, sign * s2))
            else:
                resolved.append((dep, sign))
        return resolved

    # target name -> defining value node, from the flows/derivatives mapping dict.
    # That dict may be a return literal (`return {...}, {...}`) OR assigned to a
    # local first (`derivatives = {...}; return derivatives, flows`), so we scan
    # every dict literal. We skip entries whose value is itself a dict/constant —
    # that is the module-level VARIABLES metadata spec, not a signed expression.
    target_values: Dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                    continue
                if _short_var_segment(key_node.value) not in published:
                    continue
                if isinstance(value_node, (ast.Dict, ast.Constant)):
                    continue  # VARIABLES metadata spec, not a derivative/flow expression
                target_values.setdefault(key_node.value, value_node)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id in ("flows", "derivatives", "auxiliaries"):
                    sl = tgt.slice
                    if isinstance(sl, ast.Index):
                        sl = sl.value
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and _short_var_segment(sl.value) in published:
                        target_values.setdefault(sl.value, node.value)

    out: Dict[str, Dict[str, int]] = {}

    def _accumulate(target_short: str, base: List[Tuple[str, int]], seed: frozenset) -> None:
        bucket = out.setdefault(target_short, {})
        for dep, sign in _resolve(base, seed, target_short):
            dep_short = _short_var_segment(dep)
            if dep_short == target_short:
                continue
            bucket[dep_short] = bucket.get(dep_short, 0) + sign

    for target, value_node in target_values.items():
        # Force one level through a bare local name (the dict value), else read the expression.
        if isinstance(value_node, ast.Name) and value_node.id in local_assign:
            base = local_assign[value_node.id]
            seed = frozenset({value_node.id})
        else:
            base = _extract_signed_dependencies(value_node)
            seed = frozenset()
        _accumulate(_short_var_segment(target), base, seed)

    # Auxiliaries need not travel through a flows/derivatives mapping: XMILE-generated
    # submodels publish them with `sim_context.all_params['main.x'] = x` and just
    # assign the local. Without their signs every such edge defaults to "+", which
    # silently flips a loop's polarity — `remaining_capacity_factor = 1 - state /
    # CAPACITY` reads as reinforcing rather than the balancing brake it is.
    for name, deps in local_assign.items():
        short = _short_var_segment(name)
        if short in published and short not in out:
            _accumulate(short, deps, frozenset({name}))

    return out


def _extract_signed_dependencies_from_expression(expression: str) -> List[Tuple[str, int]]:
    """Extract signed dependencies from an expression string.

    This is used as a fallback when derivative mappings reference an intermediate
    variable name (for example ``d_stock_dt``) and sign information would
    otherwise be lost.
    """
    raw = str(expression or "").strip()
    if not raw:
        return []

    rhs = raw.split("=", 1)[1].strip() if "=" in raw else raw
    if not rhs:
        return []

    try:
        parsed = ast.parse(rhs, mode="eval")
    except Exception:
        return []

    return _extract_signed_dependencies(parsed.body)


def _infer_expression_sign(node: ast.AST) -> int | None:
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            inner_sign = _infer_expression_sign(node.operand)
            return -inner_sign if inner_sign is not None else -1
        if isinstance(node.op, ast.UAdd):
            inner_sign = _infer_expression_sign(node.operand)
            return inner_sign if inner_sign is not None else 1

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left_sign = _infer_expression_sign(node.left)
            right_sign = _infer_expression_sign(node.right)
            if left_sign is not None and right_sign is not None and left_sign == right_sign:
                return left_sign
            return None
        if isinstance(node.op, ast.Sub):
            left_sign = _infer_expression_sign(node.left)
            right_sign = _infer_expression_sign(node.right)
            if left_sign is not None and right_sign is not None and left_sign == -right_sign:
                return left_sign
            return None
        if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            left_sign = _infer_expression_sign(node.left)
            right_sign = _infer_expression_sign(node.right)
            if left_sign is None or right_sign is None:
                return None
            return left_sign * right_sign

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if node.value > 0:
            return 1
        if node.value < 0:
            return -1
        return None

    if isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript)):
        return 1

    return None


def _infer_uniform_expression_sign(expression: str) -> int | None:
    raw = str(expression or "").strip()
    if not raw:
        return None

    rhs = raw.split("=", 1)[1].strip() if "=" in raw else raw
    if not rhs:
        return None

    try:
        parsed = ast.parse(rhs, mode="eval")
    except Exception:
        return None

    return _infer_expression_sign(parsed.body)


def _normalize_inline_return_mappings(tree):
    """Rewrite `return {<derivatives>}, {<flows>}` into the named-dict form.

    Generated models sometimes return the derivatives/flows mapping dicts as
    inline literals from calculate_derivatives_and_flows instead of assigning
    them to the named `derivatives` / `flows` locals first (regression fixture:
    backend/evals/fixtures/structure_recovery/overshoot_6270879390498816).
    Dependency/sign extraction keys off those names, so inline returns silently
    dropped flow->stock derivative edges. Normalizing here makes both styles
    equivalent for every downstream scan. Mirrored in
    pyodide_frontend_overlay._normalize_inline_return_mappings for the
    generated frontend bundle.
    """
    _names = ('derivatives', 'flows', 'auxiliaries')

    def _rewrite_body(body):
        new_body = []
        changed = False
        for stmt in body:
            for attr in ('body', 'orelse', 'finalbody'):
                child = getattr(stmt, attr, None)
                if isinstance(child, list) and child:
                    rewritten, child_changed = _rewrite_body(child)
                    if child_changed:
                        setattr(stmt, attr, rewritten)
                        changed = True
            if isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    rewritten, child_changed = _rewrite_body(handler.body)
                    if child_changed:
                        handler.body = rewritten
                        changed = True
            if (isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple)
                    and any(isinstance(e, ast.Dict) for e in stmt.value.elts)):
                new_elts = []
                for idx, elt in enumerate(stmt.value.elts):
                    if idx < len(_names) and isinstance(elt, ast.Dict):
                        assign = ast.Assign(
                            targets=[ast.Name(id=_names[idx], ctx=ast.Store())],
                            value=elt,
                        )
                        ast.copy_location(assign, stmt)
                        new_body.append(assign)
                        name_ref = ast.Name(id=_names[idx], ctx=ast.Load())
                        ast.copy_location(name_ref, elt)
                        new_elts.append(name_ref)
                        changed = True
                    else:
                        new_elts.append(elt)
                stmt.value.elts = new_elts
            new_body.append(stmt)
        return new_body, changed

    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == 'calculate_derivatives_and_flows':
            rewritten, changed = _rewrite_body(fn.body)
            if changed:
                fn.body = rewritten
    ast.fix_missing_locations(tree)
    return tree


def _extract_variables_from_source(file_path: str, source_code: str) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str]]]:
    namespace = normalize_namespace(file_path)
    variables: Dict[str, Dict[str, Any]] = {}
    dependencies: List[Tuple[str, str, str]] = []

    try:
        tree = _normalize_inline_return_mappings(ast.parse(source_code))
    except Exception:
        return variables, dependencies

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "VARIABLES" not in targets or not isinstance(node.value, ast.Dict):
            continue

        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            var_name = key_node.value
            if not isinstance(value_node, ast.Dict):
                continue

            var_def: Dict[str, Any] = {}
            for k_node, v_node in zip(value_node.keys, value_node.values):
                if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
                    continue
                key = k_node.value
                try:
                    var_def[key] = ast.literal_eval(v_node)
                except Exception:
                    if isinstance(v_node, ast.Constant):
                        var_def[key] = v_node.value

            variables[var_name] = {
                "id": f"{namespace}::{var_name}",
                "name": var_name,
                "namespace": namespace,
                "type": normalize_category(var_def.get("category")),
                "label": var_def.get("label") or var_name,
                "inflows": var_def.get("inflows") if isinstance(var_def.get("inflows"), list) else [],
                "outflows": var_def.get("outflows") if isinstance(var_def.get("outflows"), list) else [],
            }

            equation = var_def.get("equation")
            if isinstance(equation, str) and equation.strip():
                for raw_token in TOKEN_PATTERN.findall(equation):
                    dep_name = normalize_token(raw_token)
                    if dep_name and dep_name != var_name:
                        dependencies.append((dep_name, var_name, "signal"))

    return variables, dependencies


def build_flow_diagram_from_source_map(file_map: Dict[str, str]) -> Dict[str, Any]:
    variables: Dict[str, Dict[str, Any]] = {}
    dependencies: List[Tuple[str, str, str]] = []
    parser = VeydraModelASTParser()
    per_file_results: List[Dict[str, Any]] = []
    inferred_stock_inflows: Dict[str, Set[str]] = {}
    inferred_stock_outflows: Dict[str, Set[str]] = {}
    inferred_stock_signals: Dict[str, Dict[str, int]] = {}
    # target_name -> {source_name: summed_sign} for computational-edge polarity.
    computational_signs: Dict[str, Dict[str, int]] = {}

    for file_path, source_code in file_map.items():
        file_vars, file_deps = _extract_variables_from_source(file_path, source_code)
        variables.update(file_vars)
        dependencies.extend(file_deps)

        parse_result = parser.parse_source(source_code, file_path=file_path)
        per_file_results.append({
            "file_path": file_path,
            "result": parse_result,
        })

        namespace = normalize_namespace(file_path)
        stocks = set(parse_result.get("stocks", {}).keys())
        flows = set(parse_result.get("flows", {}).keys())
        parameters = set(parse_result.get("parameters", {}).keys())
        auxiliary = set(parse_result.get("auxiliary", {}).keys())

        for var_name, var_def in parse_result.get("variables", {}).items():
            inferred_type = "auxiliary"
            if var_name in stocks:
                inferred_type = "stock"
            elif var_name in flows:
                inferred_type = "flow"
            elif var_name in parameters:
                inferred_type = "parameter"
            elif var_name in auxiliary:
                inferred_type = normalize_category(var_def.get("category"))

            existing = variables.get(var_name, {})
            inflows = var_def.get("inflows") if isinstance(var_def.get("inflows"), list) else existing.get("inflows", [])
            outflows = var_def.get("outflows") if isinstance(var_def.get("outflows"), list) else existing.get("outflows", [])
            variables[var_name] = {
                "id": existing.get("id") or f"{namespace}::{var_name}",
                "name": var_name,
                "namespace": existing.get("namespace") or namespace,
                "type": existing.get("type") or inferred_type,
                "label": var_def.get("name") or var_def.get("label") or existing.get("label") or var_name,
                "inflows": inflows,
                "outflows": outflows,
            }

    # Auxiliaries VeydraModelASTParser finds in code but that VARIABLES never
    # declares are still real intermediate nodes. Without them every causal chain
    # routing through such a local collapses onto its endpoints, so in
    # `growth_flow = biomass * effective_growth_rate` (with the logistic brake
    # living inside effective_growth_rate) the positive direct effect and the
    # negative brake merge into one biomass -> growth_flow edge whose signs
    # cancel to "+", hiding the balancing loop. Publishing the aux splits them:
    # _resolve_computational_signs_from_tree stops expanding at any published
    # name, so the aux gets its own sign bucket.
    published_aliases = {
        candidate for var_name in variables for candidate in alias_candidates(var_name)
    }
    for file_result in per_file_results:
        namespace = normalize_namespace(file_result["file_path"])
        for aux_name, aux_def in (file_result["result"].get("auxiliary", {}) or {}).items():
            if aux_name in variables or aux_name in published_aliases:
                continue  # a declared variable already owns this name (e.g. a param's local)
            variables[aux_name] = {
                "id": f"{namespace}::{aux_name}",
                "name": aux_name,
                "namespace": namespace,
                "type": "auxiliary",
                "label": aux_def.get("name") or aux_name,
                "inflows": [],
                "outflows": [],
            }
            published_aliases.update(alias_candidates(aux_name))

    alias_index = build_alias_index(variables)

    # A flow defined as a bare local (`drain_dt = actual_drain_flow`) *is* that
    # local: let the local resolve to the flow so derivative expressions written
    # in terms of the local still reach the flow node. Additive — a local that is
    # also a published variable keeps resolving to that variable too.
    published_shorts = {_short_var_segment(name) for name in variables}
    flow_alias_short: Dict[str, str] = {}
    for file_result in per_file_results:
        flow_expressions = file_result["result"].get("flow_expressions", {}) or {}
        for local, flow_names in _flow_identity_aliases(flow_expressions, variables).items():
            alias_index.setdefault(local, set()).update(flow_names)
            local_short = _short_var_segment(local)
            if local_short in published_shorts or len(flow_names) != 1:
                continue  # ambiguous, or the local names a real variable already
            flow_alias_short[local_short] = _short_var_segment(next(iter(flow_names)))

    for file_result in per_file_results:
        parse_result = file_result["result"]
        source_code = file_map.get(file_result["file_path"], "")
        try:
            file_dependencies = _merge_variable_dependencies(
                parse_result.get("variable_dependencies", {}) or {},
                _collect_local_variable_dependencies(
                    _normalize_inline_return_mappings(ast.parse(source_code))
                ),
            )
        except Exception:
            file_dependencies = parse_result.get("variable_dependencies", {}) or {}

        published_targets = set()
        published_targets.update(parse_result.get("flows", {}).keys())
        published_targets.update(parse_result.get("stocks", {}).keys())
        published_targets.update(parse_result.get("auxiliary", {}).keys())

        for target_name in published_targets:
            exclude = {target_name}
            for target_alias in alias_candidates(target_name):
                for dep_name in file_dependencies.get(target_alias, []):
                    for source_name in resolve_dependency_targets(
                        dep_name, file_dependencies, alias_index, None, exclude
                    ):
                        if source_name and source_name != target_name:
                            dependencies.append((source_name, target_name, "computational"))

        try:
            tree = _normalize_inline_return_mappings(ast.parse(source_code))
            target_ids = set(parse_result.get("variables", {}).keys())
            signed_deps = _extract_target_signed_dependencies_from_tree(tree, target_ids)
            # Real +/- polarity for computational edges (instead of a hardcoded "+"),
            # following local assignment chains. Keyed by short name so it matches
            # edges regardless of namespace prefixing.
            _published_short = {_short_var_segment(k) for k in variables.keys()}
            _published_short.update(_short_var_segment(k) for k in target_ids)
            for _tgt_short, _dep_signs in _resolve_computational_signs_from_tree(
                tree, _published_short, flow_alias_short
            ).items():
                computational_signs.setdefault(_tgt_short, {})
                for _dep_short, _sign in _dep_signs.items():
                    computational_signs[_tgt_short][_dep_short] = (
                        computational_signs[_tgt_short].get(_dep_short, 0) + _sign)
            stock_expressions = parse_result.get("stock_expressions", {}) or {}

            for stock_name, stock_info in variables.items():
                if stock_info.get("type") != "stock":
                    continue

                base_signed_deps: List[Tuple[str, int]] = list(signed_deps.get(stock_name, []))

                stock_expression = stock_expressions.get(stock_name)
                expression_signed_deps = _extract_signed_dependencies_from_expression(stock_expression)

                if expression_signed_deps:
                    # If we have a concrete stock derivative expression (e.g.
                    # `d_stock_dt = inflow - outflow`), trust it over alias-based
                    # signed deps to avoid double-counting through `d_*_dt` symbols.
                    merged_signed_deps = []
                    for dep_name, dep_sign in expression_signed_deps:
                        pair = (dep_name, dep_sign)
                        if pair not in merged_signed_deps:
                            merged_signed_deps.append(pair)

                    # Preserve direct flow aliases from the derivative mapping
                    # (for example `{'stock': price_change_flow}`) so physical
                    # stock-flow links survive when the alias expands to a raw
                    # arithmetic formula with no published flow identifier left.
                    expression_sign = _infer_uniform_expression_sign(stock_expression)
                    for dep_name, dep_sign in base_signed_deps:
                        if _resolves_to_flow(dep_name, file_dependencies, alias_index, variables, {stock_name}):
                            resolved_sign = expression_sign if expression_sign is not None else dep_sign
                            pair = (dep_name, resolved_sign)
                            if pair in merged_signed_deps:
                                continue
                            merged_signed_deps.append(pair)
                else:
                    merged_signed_deps = base_signed_deps

                # Prefer signs traced through local assignments. When the stock
                # derivative is a local (`{stock: d_stock_dt}` with `d_stock_dt =
                # inflow - outflow`), the alias resolution below loses the minus
                # sign and both flows read as inflows. computational_signs already
                # expanded that assignment with correct per-flow signs.
                stock_short = _short_var_segment(stock_name)
                comp_flow_signs = computational_signs.get(stock_short, {})
                resolved_from_comp: Set[str] = set()
                if comp_flow_signs:
                    for _vname, _vinfo in variables.items():
                        if _vinfo.get("type") != "flow":
                            continue
                        _sign = comp_flow_signs.get(_short_var_segment(_vname))
                        if _sign:
                            inferred_stock_signals.setdefault(stock_name, {})
                            inferred_stock_signals[stock_name][_vname] = _sign
                            resolved_from_comp.add(_vname)

                # Pre-seed `visited` with the stock's own alias names: `exclude`
                # stops the stock terminating a walk, but must not let the walk
                # pass THROUGH the stock's dependency bucket (stock -> its own
                # derivative flow) — that routes a stock-reading dep like
                # `s_price` back onto the flow with a spurious +1 that cancels
                # the real signed contribution.
                _stock_alias_block = set(alias_candidates(stock_name)) | {stock_name}
                for dep_name, dep_sign in merged_signed_deps:
                    for source_name in resolve_dependency_targets(
                        dep_name, file_dependencies, alias_index, set(_stock_alias_block), {stock_name}
                    ):
                        if variables.get(source_name, {}).get("type") != "flow":
                            continue
                        if source_name in resolved_from_comp:
                            continue  # authoritative sign already set from local-assignment tracing

                        inferred_stock_signals.setdefault(stock_name, {})
                        inferred_stock_signals[stock_name][source_name] = inferred_stock_signals[stock_name].get(source_name, 0) + dep_sign
        except Exception:
            pass

    for stock_name, flow_signals in inferred_stock_signals.items():
        for flow_name, sign_total in flow_signals.items():
            if sign_total > 0:
                inferred_stock_inflows.setdefault(stock_name, set()).add(flow_name)
            elif sign_total < 0:
                inferred_stock_outflows.setdefault(stock_name, set()).add(flow_name)

    edges_seen = set()
    edges: List[Dict[str, Any]] = []

    for source_name, target_name, rel_type in dependencies:
        if source_name not in variables or target_name not in variables:
            continue
        source_id = variables[source_name]["id"]
        target_id = variables[target_name]["id"]
        edge_key = (source_id, target_id, rel_type)
        if edge_key in edges_seen:
            continue
        edges_seen.add(edge_key)
        polarity = "+"
        if rel_type == "computational":
            target_signs = computational_signs.get(_short_var_segment(target_name), {})
            sign = target_signs.get(_short_var_segment(source_name))
            if sign is not None and sign < 0:
                polarity = "-"
        edges.append(
            {
                "id": f"{source_id}->{target_id}:{rel_type}",
                "source": source_id,
                "target": target_id,
                "data": {"relationship_type": rel_type, "polarity": polarity},
            }
        )

    for var_name, info in variables.items():
        if info.get("type") != "stock":
            continue
        stock_id = info["id"]
        explicit_inflows = {str(name) for name in info.get("inflows", [])}
        explicit_outflows = {str(name) for name in info.get("outflows", [])}
        inferred_inflows = inferred_stock_inflows.get(var_name, set())
        inferred_outflows = inferred_stock_outflows.get(var_name, set())

        # Keep explicit declarations, but union with inferred signs from stock
        # expressions so stale/partial VARIABLES metadata cannot hide real links.
        inflow_names = explicit_inflows | inferred_inflows
        outflow_names = explicit_outflows | inferred_outflows

        for flow_name in sorted(inflow_names):
            flow = resolve_flow_variable(str(flow_name), var_name, info, variables)
            if not flow:
                continue
            flow_id = flow["id"]
            edge_key = (flow_id, stock_id, "physical")
            if edge_key in edges_seen:
                continue
            edges_seen.add(edge_key)
            edges.append(
                {
                    "id": f"{flow_id}->{stock_id}:physical",
                    "source": flow_id,
                    "target": stock_id,
                    "data": {"relationship_type": "physical", "polarity": "+"},
                }
            )

        for flow_name in sorted(outflow_names):
            flow = resolve_flow_variable(str(flow_name), var_name, info, variables)
            if not flow:
                continue
            flow_id = flow["id"]
            edge_key = (stock_id, flow_id, "physical")
            if edge_key in edges_seen:
                continue
            edges_seen.add(edge_key)
            edges.append(
                {
                    "id": f"{stock_id}->{flow_id}:physical",
                    "source": stock_id,
                    "target": flow_id,
                    "data": {"relationship_type": "physical", "polarity": "+"},
                }
            )

    flow_has_source: Dict[str, bool] = {}
    flow_has_target: Dict[str, bool] = {}
    for var_name, info in variables.items():
        if info.get("type") == "flow":
            flow_has_source[var_name] = False
            flow_has_target[var_name] = False

    for edge in edges:
        edge_type = edge.get("data", {}).get("relationship_type")
        if edge_type != "physical":
            continue

        source_id = edge.get("source")
        target_id = edge.get("target")
        source_name = next((name for name, info in variables.items() if info.get("id") == source_id), None)
        target_name = next((name for name, info in variables.items() if info.get("id") == target_id), None)

        if source_name and variables.get(source_name, {}).get("type") == "stock" and target_name and variables.get(target_name, {}).get("type") == "flow":
            flow_has_source[target_name] = True
        if source_name and variables.get(source_name, {}).get("type") == "flow" and target_name and variables.get(target_name, {}).get("type") == "stock":
            flow_has_target[source_name] = True

    for flow_name, info in list(variables.items()):
        if info.get("type") != "flow":
            continue

        has_source = flow_has_source.get(flow_name, False)
        has_target = flow_has_target.get(flow_name, False)
        if not (has_source ^ has_target):
            continue

        namespace = info.get("namespace") or "main"
        flow_domain = flow_name.rsplit(".", 1)[0] if "." in flow_name else namespace
        flow_base = boundary_flow_base_name(flow_name)

        if has_target and not has_source:
            cloud_name = f"{flow_domain}.{flow_base}_source" if flow_domain else f"{flow_base}_source"
            cloud_id = f"{namespace}::{cloud_name}" if namespace else cloud_name
            if cloud_name not in variables:
                variables[cloud_name] = {
                    "id": cloud_id,
                    "name": cloud_name,
                    "namespace": namespace,
                    "type": "cloud",
                    "label": cloud_name,
                    "inflows": [],
                    "outflows": [],
                }

            edge_key = (cloud_id, info["id"], "physical")
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                edges.append(
                    {
                        "id": f"{cloud_id}->{info['id']}:physical",
                        "source": cloud_id,
                        "target": info["id"],
                        "data": {"relationship_type": "physical", "polarity": "+", "context": "cloud_source"},
                    }
                )

        if has_source and not has_target:
            cloud_name = f"{flow_domain}.{flow_base}_sink" if flow_domain else f"{flow_base}_sink"
            cloud_id = f"{namespace}::{cloud_name}" if namespace else cloud_name
            if cloud_name not in variables:
                variables[cloud_name] = {
                    "id": cloud_id,
                    "name": cloud_name,
                    "namespace": namespace,
                    "type": "cloud",
                    "label": cloud_name,
                    "inflows": [],
                    "outflows": [],
                }

            edge_key = (info["id"], cloud_id, "physical")
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                edges.append(
                    {
                        "id": f"{info['id']}->{cloud_id}:physical",
                        "source": info["id"],
                        "target": cloud_id,
                        "data": {"relationship_type": "physical", "polarity": "+", "context": "cloud_sink"},
                    }
                )

    type_order = ["stock", "flow", "parameter", "auxiliary", "constant", "cloud"]
    type_column = {var_type: idx for idx, var_type in enumerate(type_order)}
    type_counters = {var_type: 0 for var_type in type_order}

    node_positions: Dict[str, Dict[str, float]] = {}
    nodes: List[Dict[str, Any]] = []
    for var_name in sorted(variables.keys()):
        info = variables[var_name]
        node_type = info.get("type") or "auxiliary"
        if node_type not in type_column:
            node_type = "auxiliary"

        if node_type == "cloud":
            continue

        idx = type_counters.get(node_type, 0)
        type_counters[node_type] = idx + 1

        position = {
            "x": 140 + (type_column[node_type] * 230),
            "y": 120 + (idx * 95),
        }
        node_positions[info["id"]] = position

        nodes.append(
            {
                "id": info["id"],
                "position": position,
                "data": {
                    "label": info.get("label") or var_name,
                    "type": node_type,
                    "namespace": info.get("namespace") or "main",
                    "variable_name": var_name,
                },
            }
        )

    cloud_offset_x = 170
    cloud_offset_y = 120
    fallback_cloud_index = 0
    for var_name in sorted(variables.keys()):
        info = variables[var_name]
        if info.get("type") != "cloud":
            continue

        base_name = str(var_name).split(".")[-1]
        namespace = info.get("namespace") or "main"
        is_source = base_name.endswith("_source")
        is_sink = base_name.endswith("_sink")
        if is_source:
            flow_stem = base_name[:-7]
        elif is_sink:
            flow_stem = base_name[:-5]
        else:
            flow_stem = base_name

        flow_id = None
        for suffix in ("_flow", "_rate", ""):
            candidate_name = f"{namespace}.{flow_stem}{suffix}" if namespace else f"{flow_stem}{suffix}"
            candidate = variables.get(candidate_name)
            if candidate and candidate.get("id") in node_positions:
                flow_id = candidate.get("id")
                break

        if flow_id and flow_id in node_positions:
            flow_position = node_positions[flow_id]
            if is_source:
                position = {
                    "x": flow_position["x"] - cloud_offset_x,
                    "y": flow_position["y"] - cloud_offset_y,
                }
            elif is_sink:
                position = {
                    "x": flow_position["x"] + cloud_offset_x,
                    "y": flow_position["y"] + cloud_offset_y,
                }
            else:
                position = {
                    "x": flow_position["x"],
                    "y": flow_position["y"] - cloud_offset_y,
                }
        else:
            position = {
                "x": 140 + (fallback_cloud_index * 150),
                "y": 30,
            }
            fallback_cloud_index += 1

        node_positions[info["id"]] = position
        nodes.append(
            {
                "id": info["id"],
                "position": position,
                "data": {
                    "label": info.get("label") or var_name,
                    "type": "cloud",
                    "namespace": namespace,
                    "variable_name": var_name,
                },
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "parser": "runtime_diagram_parser",
            "version": "v1",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }


# Edge types that carry causal influence, and so can form a feedback loop.
# "computational" is what build_flow_diagram_from_source_map emits for every
# dependency it reads out of calculate_derivatives_and_flows — omitting it left
# loop detection with only the acyclic stock/flow skeleton and zero loops.
CAUSAL_RELATIONSHIP_TYPES = ("causal", "computational", "mathematical", "physical", "signal")


def _node_types_from_flow_diagram(flow_diagram: Dict[str, Any]) -> Dict[str, str]:
    """Map node id -> declared type ('stock' | 'flow' | ...) from diagram nodes."""
    node_types: Dict[str, str] = {}
    for node in flow_diagram.get("nodes", []) or []:
        if isinstance(node, dict) and node.get("id"):
            node_types[str(node["id"])] = str(
                ((node.get("data") or {}).get("type")) or ""
            ).strip().lower()
    return node_types


def build_feedback_loops_from_flow_diagram(flow_diagram: Dict[str, Any], max_loop_length: int = 8) -> Dict[str, Any]:
    edges = flow_diagram.get("edges", [])
    causal_links: List[Dict[str, Any]] = []
    node_types = _node_types_from_flow_diagram(flow_diagram)

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        edge_data = edge.get("data", {})
        relationship_type = edge_data.get("relationship_type", "unknown")
        polarity = edge_data.get("polarity", "+")

        # A physical stock->flow edge is the visual outflow pipe — conservation
        # plumbing, not causation. Counting it as a causal link fabricates a
        # balancing loop for every outflow that does NOT read its stock (e.g.
        # depletion_flow = pop * cons_rate draining s_resource_levels). When the
        # flow genuinely depends on the stock, the signed COMPUTATIONAL edge
        # with the same endpoints carries the link, so dropping the pipe loses
        # nothing. Flow->stock physical edges stay: they mirror the derivative
        # and carry the conserved quantity into the stock.
        if (
            relationship_type == "physical"
            and node_types.get(str(source)) == "stock"
            and node_types.get(str(target)) == "flow"
        ):
            continue

        if relationship_type in CAUSAL_RELATIONSHIP_TYPES and source and target:
            causal_links.append(
                {
                    "source": source,
                    "target": target,
                    "polarity": polarity,
                    "relationship_type": relationship_type,
                    "edge_id": edge.get("id", f"{source}-{target}"),
                    "evidence": edge_data.get("evidence", ""),
                    "context": edge_data.get("context", "runtime_relationship"),
                }
            )

    loop_analysis = find_feedback_loops(causal_links, max_loop_length=max_loop_length)
    enhanced_loops: List[Dict[str, Any]] = []

    for i, loop in enumerate(loop_analysis.get("loops", []), 1):
        namespaces = sorted({v.split("::", 1)[0] for v in loop.get("variables", []) if "::" in v})
        enhanced_loops.append(
            {
                **loop,
                "id": f"loop_{i}",
                "namespaces": namespaces,
                "namespace_count": len(namespaces),
                "crosses_boundaries": len(namespaces) > 1,
            }
        )

    loop_analysis["loops"] = enhanced_loops
    loop_analysis["namespace_analysis"] = {
        "total_namespaces": len({ns for loop in enhanced_loops for ns in loop.get("namespaces", [])}),
        "cross_boundary_count": sum(1 for loop in enhanced_loops if loop.get("crosses_boundaries")),
    }

    cld_data = create_cld_data_structure(causal_links, loop_analysis)

    return {
        "metadata": {
            "analysis_type": "feedback_loops",
            "version": "runtime-v1",
            "source": "runtime_diagram_parser",
        },
        "loop_analysis": {
            **loop_analysis,
            "metadata": {
                "total_loops": loop_analysis.get("total_loops", 0),
                "reinforcing_loops": loop_analysis.get("reinforcing_loops", 0),
                "balancing_loops": loop_analysis.get("balancing_loops", 0),
                "total_variables": cld_data.get("statistics", {}).get("total_variables", 0),
                "total_relationships": cld_data.get("statistics", {}).get("total_relationships", 0),
            },
        },
        "causal_loop_diagram": {
            "nodes": [
                {"id": v["name"], "type": "variable", "label": v["name"]}
                for v in cld_data.get("variables", [])
            ],
            "edges": [
                {
                    "source": r.get("source"),
                    "target": r.get("target"),
                    "polarity": r.get("polarity", "+"),
                }
                for r in causal_links
            ],
        },
        "visualization": {
            "color_scheme": {
                "positive_polarity": "green",
                "negative_polarity": "red",
            }
        },
    }


def build_flow_and_feedback_from_source_map(file_map: Dict[str, str], max_loop_length: int = 8) -> Dict[str, Any]:
    flow_diagram = build_flow_diagram_from_source_map(file_map)
    feedback_loops = build_feedback_loops_from_flow_diagram(flow_diagram, max_loop_length=max_loop_length)
    return {
        "flow_diagram": flow_diagram,
        "feedback_loops": feedback_loops,
    }
