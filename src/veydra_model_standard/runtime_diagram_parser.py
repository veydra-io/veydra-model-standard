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
) -> Set[str]:
    if not dependency_name:
        return set()

    if visited is None:
        visited = set()

    if dependency_name in visited:
        return set()

    visited.add(dependency_name)

    resolved = set(alias_index.get(dependency_name, set()))
    for child_dependency in file_dependencies.get(dependency_name, []):
        resolved.update(resolve_dependency_targets(child_dependency, file_dependencies, alias_index, visited))
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


def _extract_variables_from_source(file_path: str, source_code: str) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str]]]:
    namespace = normalize_namespace(file_path)
    variables: Dict[str, Dict[str, Any]] = {}
    dependencies: List[Tuple[str, str, str]] = []

    try:
        tree = ast.parse(source_code)
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

    alias_index = build_alias_index(variables)

    for file_result in per_file_results:
        parse_result = file_result["result"]
        file_dependencies = parse_result.get("variable_dependencies", {}) or {}
        source_code = file_map.get(file_result["file_path"], "")

        published_targets = set()
        published_targets.update(parse_result.get("flows", {}).keys())
        published_targets.update(parse_result.get("stocks", {}).keys())
        published_targets.update(parse_result.get("auxiliary", {}).keys())

        for target_name in published_targets:
            for target_alias in alias_candidates(target_name):
                for source_name in resolve_dependency_targets(target_alias, file_dependencies, alias_index):
                    if source_name and source_name != target_name:
                        dependencies.append((source_name, target_name, "computational"))

        try:
            tree = ast.parse(source_code)
            target_ids = set(parse_result.get("variables", {}).keys())
            signed_deps = _extract_target_signed_dependencies_from_tree(tree, target_ids)
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
                else:
                    merged_signed_deps = base_signed_deps

                for dep_name, dep_sign in merged_signed_deps:
                    for source_name in resolve_dependency_targets(dep_name, file_dependencies, alias_index):
                        if variables.get(source_name, {}).get("type") != "flow":
                            continue

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
        edges.append(
            {
                "id": f"{source_id}->{target_id}:{rel_type}",
                "source": source_id,
                "target": target_id,
                "data": {"relationship_type": rel_type, "polarity": "+"},
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


def build_feedback_loops_from_flow_diagram(flow_diagram: Dict[str, Any], max_loop_length: int = 8) -> Dict[str, Any]:
    edges = flow_diagram.get("edges", [])
    causal_links: List[Dict[str, Any]] = []

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        edge_data = edge.get("data", {})
        relationship_type = edge_data.get("relationship_type", "unknown")
        polarity = edge_data.get("polarity", "+")

        if relationship_type in ("causal", "mathematical", "physical", "signal") and source and target:
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
