"""
AST Parser for Veydra Model Standard (VMS) Python files.

This module provides functions to parse Python source code using AST
and extract parameters, stocks, flows, and auxiliary variables according
to the Veydra Model Standard defined in veydra_model_standard.py.
"""

import ast
import inspect
import re
import sys
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime


class VeydraModelASTParser:
    """
    AST parser for extracting Veydra model components from Python source code.
    
    Extracts:
    - Parameters: Configuration values with defaults, types, and validation
    - Stocks: State variables that accumulate over time (category='stock')
    - Flows: Rate variables that represent changes (category='rate' or _dt suffix)
    - Auxiliary variables: Computed intermediate values
    - Variable metadata: names, units, descriptions, categories, validation rules
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset internal state for parsing a new file."""
        self.variables = {}
        self.parameters = {}
        self.stocks = {}
        self.flows = {}
        self.flow_expressions = {}
        self.flow_expression_sources = {}
        self.flow_function_sources = {}
        self.stock_expressions = {}
        self.stock_expression_sources = {}
        self.stock_function_sources = {}
        self.auxiliary_expressions = {}
        self.auxiliary_expression_sources = {}
        self.auxiliary_function_sources = {}
        self.calc_function_definitions = {}
        self.auxiliary = {}
        self.class_methods = {}
        self.imports = []
        self.submodels = []
        # VMS Single-Assignment Convention tracking
        self.duplicate_assignments = []  # Warnings for variables assigned multiple times
        self.variable_dependencies = {}  # Maps variable -> list of dependencies
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Python file and extract all VMS components.
        
        Args:
            file_path: Path to the Python file to parse
            
        Returns:
            Dictionary containing extracted components:
            {
                'variables': dict,      # All VARIABLES dict content
                'parameters': dict,     # Parameters only (not stocks/flows)
                'stocks': dict,         # Stock variables
                'flows': dict,          # Flow variables
                'auxiliary': dict,      # Auxiliary/computed variables
                'submodels': list,      # Detected submodel classes
                'imports': list,        # Import statements
                'methods': dict,        # Class methods by class name
                'metadata': dict        # File-level metadata
            }
        """
        self.reset()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return self.parse_source(source_code, file_path)
    
    def parse_source(self, source_code: str, file_path: str = "<string>") -> Dict[str, Any]:
        """
        Parse Python source code and extract VMS components.
        
        Args:
            source_code: Python source code as string
            file_path: Optional file path for error reporting
            
        Returns:
            Dictionary containing extracted components
        """
        self.reset()
        
        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            return {
                'error': f"Syntax error in {file_path}: {e}",
                'variables': {},
                'parameters': {},
                'stocks': {},
                'flows': {},
                'auxiliary': {},
                'submodels': [],
                'imports': [],
                'methods': {},
                'metadata': {'file_path': file_path}
            }
        
        # Extract components
        self._extract_imports(tree)
        self._extract_variables_dict(tree)

        # Runtime fallback: if VARIABLES was initially assigned as an empty dict
        # (i.e. ``VARIABLES = {}``) it probably means the real entries are being
        # built dynamically via loops or helper functions that AST can't resolve.
        # In that case, import the module at runtime to capture the full dict.
        if self._variables_initialized_empty(tree):
            self._try_runtime_variables_import(file_path)

        self._extract_classes_and_methods(tree)
        self._categorize_variables()
        self._extract_dynamic_flows_from_methods(tree)  # NEW: Extract flows from calculate_derivatives_and_flows
        self._extract_auxiliary_variables(tree)
        self._extract_flow_expressions(tree)
        self._extract_stock_expressions(tree)
        self._extract_auxiliary_expressions(tree)
        self._extract_calc_function_definitions(tree, source_code, file_path)
        
        return {
            'variables': self.variables,
            'parameters': self.parameters,
            'stocks': self.stocks,
            'flows': self.flows,
            'flow_expressions': self.flow_expressions,
            'flow_expression_sources': self.flow_expression_sources,
            'flow_function_sources': self.flow_function_sources,
            'stock_expressions': self.stock_expressions,
            'stock_expression_sources': self.stock_expression_sources,
            'stock_function_sources': self.stock_function_sources,
            'auxiliary_expressions': self.auxiliary_expressions,
            'auxiliary_expression_sources': self.auxiliary_expression_sources,
            'auxiliary_function_sources': self.auxiliary_function_sources,
            'calc_function_definitions': self.calc_function_definitions,
            'auxiliary': self.auxiliary,
            'submodels': self.submodels,
            'imports': self.imports,
            'methods': self.class_methods,
            'variable_dependencies': self.variable_dependencies,  # VMS: var -> [dependencies]
            'duplicate_assignments': self.duplicate_assignments,   # VMS: convention warnings
            'metadata': {
                'file_path': file_path,
                'total_variables': len(self.variables),
                'parameters_count': len(self.parameters),
                'stocks_count': len(self.stocks),
                'flows_count': len(self.flows),
                'auxiliary_count': len(self.auxiliary),
                'submodels_count': len(self.submodels),
                'duplicate_assignment_warnings': len(self.duplicate_assignments)
            }
        }
    
    def _extract_imports(self, tree: ast.AST):
        """Extract import statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    self.imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
    
    def _extract_variables_dict(self, tree: ast.AST):
        """Extract the VARIABLES dictionary definition and other variable dictionaries.

        Handles three patterns:
          1. VARIABLES = { ... }          -- direct dict literal assignment
          2. VARIABLES.update({ ... })    -- dict literal passed to .update()
          3. SIMULATION_VARIABLES = { ... }
          4. AUXILIARY_VARIABLES = { ... }
        """
        variable_dict_names = {'VARIABLES', 'SIMULATION_VARIABLES', 'AUXILIARY_VARIABLES'}
        for node in ast.walk(tree):
            # Pattern 1, 3 & 4: Direct dict assignment
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in variable_dict_names:
                        if isinstance(node.value, ast.Dict):
                            self.variables.update(self._parse_dict_node(node.value))

            # Pattern 2: VARIABLES.update({ ... })
            # AST shape: Expr(Call(func=Attribute(value=Name('VARIABLES'), attr='update'), args=[Dict(...)]))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (isinstance(call.func, ast.Attribute)
                        and call.func.attr == 'update'
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id in variable_dict_names
                        and call.args
                        and isinstance(call.args[0], ast.Dict)):
                    self.variables.update(self._parse_dict_node(call.args[0]))

    @staticmethod
    def _variables_initialized_empty(tree: ast.AST) -> bool:
        """Check whether VARIABLES was assigned as an empty dict ``VARIABLES = {}``."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name)
                            and target.id == 'VARIABLES'
                            and isinstance(node.value, ast.Dict)
                            and len(node.value.keys) == 0):
                        return True
        return False

    def _try_runtime_variables_import(self, file_path: str):
        """Fallback: import the module at runtime and read its VARIABLES dict.

        This handles submodels that build VARIABLES dynamically (loops, helper
        functions, etc.) which are invisible to static AST analysis.  We import
        the module in an isolated manner, read the VARIABLES dict, and convert
        each entry into the same shape the AST parser would have produced.
        """
        import importlib
        import importlib.util

        file_path = str(file_path)
        try:
            # Add the file's directory to sys.path temporarily so sibling
            # imports (e.g. ``from supply_chain_dynamics import ...``) work.
            file_dir = str(Path(file_path).parent)
            path_added = file_dir not in sys.path
            if path_added:
                sys.path.insert(0, file_dir)

            module_name = Path(file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            variables_dict = getattr(mod, 'VARIABLES', None)
            if variables_dict and isinstance(variables_dict, dict):
                self.variables.update(variables_dict)

            if path_added:
                sys.path.remove(file_dir)
        except Exception:
            # If import fails for any reason we silently keep whatever AST
            # managed to extract — this is just a best-effort fallback.
            pass

    def _parse_dict_node(self, dict_node: ast.Dict) -> Dict[str, Any]:
        """Recursively parse a dictionary AST node."""
        result = {}
        
        for key_node, value_node in zip(dict_node.keys, dict_node.values):
            if key_node is None:  # Handle **kwargs expansion
                continue
                
            key = self._get_literal_value(key_node)
            if key is None:
                continue
                
            if isinstance(value_node, ast.Dict):
                result[key] = self._parse_dict_node(value_node)
            else:
                result[key] = self._get_literal_value(value_node)
        
        return result
    
    def _get_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        # Handle deprecated AST nodes for backwards compatibility
        elif hasattr(ast, 'Str') and isinstance(node, ast.Str):
            return node.s
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n
        elif hasattr(ast, 'NameConstant') and isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id  # Return the name as string for variables
        elif isinstance(node, ast.List):
            return [self._get_literal_value(item) for item in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_literal_value(item) for item in node.elts)
        else:
            return None
    
    def _extract_classes_and_methods(self, tree: ast.AST):
        """Extract class definitions and their methods."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [self._get_base_class_name(base) for base in node.bases],
                    'methods': [],
                    'is_submodel': self._is_veydra_submodel(node),
                    'is_orchestrator': self._is_veydra_orchestrator(node)
                }
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = self._extract_method_info(item)
                        class_info['methods'].append(method_info)
                
                self.class_methods[node.name] = class_info
                
                # Track submodels and orchestrators
                if class_info['is_submodel'] or class_info['is_orchestrator']:
                    self.submodels.append(class_info)
    
    def _get_base_class_name(self, base_node: ast.AST) -> str:
        """Get base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return f"{self._get_base_class_name(base_node.value)}.{base_node.attr}"
        else:
            return str(base_node)
    
    def _is_veydra_submodel(self, class_node: ast.ClassDef) -> bool:
        """Check if class inherits from Submodel."""
        for base in class_node.bases:
            base_name = self._get_base_class_name(base)
            if base_name == 'Submodel':
                return True
        return False
    
    def _is_veydra_orchestrator(self, class_node: ast.ClassDef) -> bool:
        """Check if class inherits from VeydraModelStandard."""
        for base in class_node.bases:
            base_name = self._get_base_class_name(base)
            if base_name == 'VeydraModelStandard':
                return True
        return False
    
    def _extract_method_info(self, method_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract information about a class method."""
        return {
            'name': method_node.name,
            'args': [arg.arg for arg in method_node.args.args],
            'returns': self._get_return_annotation(method_node),
            'docstring': ast.get_docstring(method_node),
            'is_property': any(isinstance(d, ast.Name) and d.id == 'property' 
                             for d in method_node.decorator_list),
            'is_classmethod': any(isinstance(d, ast.Name) and d.id == 'classmethod' 
                                for d in method_node.decorator_list),
            'calls_sim_context': self._uses_sim_context(method_node),
            'line_number': method_node.lineno
        }
    
    def _get_return_annotation(self, method_node: ast.FunctionDef) -> Optional[str]:
        """Get return type annotation if present."""
        if method_node.returns:
            return ast.unparse(method_node.returns) if hasattr(ast, 'unparse') else str(method_node.returns)
        return None
    
    def _uses_sim_context(self, method_node: ast.FunctionDef) -> bool:
        """Check if method uses SimulationContext."""
        for node in ast.walk(method_node):
            if isinstance(node, ast.Name) and node.id == 'sim_context':
                return True
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == 'sim_context':
                    return True
        return False
    
    def _categorize_variables(self):
        """Categorize variables into parameters, stocks, flows based on VMS rules."""
        for var_name, var_def in self.variables.items():
            if not isinstance(var_def, dict):
                continue
                
            category = var_def.get('category', '')
            
            # Categorize based on explicit category
            if category == 'stock':
                self.stocks[var_name] = var_def
            elif category == 'rate' or category == 'flow':
                self.flows[var_name] = var_def
            elif category == 'auxiliary':
                self.auxiliary[var_name] = var_def
            elif category == 'parameter':
                self.parameters[var_name] = var_def
            # Implicit categorization based on naming conventions
            elif var_name.endswith('_dt') or 'rate' in var_name.lower():
                self.flows[var_name] = var_def
            elif var_name.startswith('s_') or 'inventory' in var_name.lower():
                # Likely a stock if it starts with 's_' or contains 'inventory'
                self.stocks[var_name] = var_def
            elif var_name.startswith('auxiliary.'):
                self.auxiliary[var_name] = var_def
            else:
                # Default to parameter if no clear category
                self.parameters[var_name] = var_def
    
    def _extract_dynamic_flows_from_methods(self, tree: ast.AST):
        """Extract flows from calculate_derivatives_and_flows() method return dicts.
        
        VMS submodels compute flows dynamically and return them as dictionary literals
        in the calculate_derivatives_and_flows() method. This method extracts those
        flow keys and their expressions.
        """
        for node in ast.walk(tree):
            # Look for class definitions (submodels)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    # Find calculate_derivatives_and_flows method
                    if isinstance(item, ast.FunctionDef) and item.name == 'calculate_derivatives_and_flows':
                        self._extract_flows_from_method(item, node.name)
    
    def _extract_flows_from_method(self, method_node: ast.FunctionDef, class_name: str):
        """Extract flow definitions from a method body."""
        for stmt in ast.walk(method_node):
            # Look for: flows = {...}
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == 'flows':
                        if isinstance(stmt.value, ast.Dict):
                            self._extract_flows_from_dict(stmt.value, class_name)
    
    def _extract_flows_from_dict(self, dict_node: ast.Dict, class_name: str):
        """Extract flow names and expressions from a dict literal."""
        for key, value in zip(dict_node.keys, dict_node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                flow_name = key.value
                # Don't overwrite if already defined in VARIABLES
                if flow_name not in self.flows:
                    self.flows[flow_name] = {
                        'name': flow_name,
                        'category': 'flow',
                        'source_class': class_name,
                        'expression': self._get_expression_string(value),
                        'dynamic': True  # Mark as dynamically computed
                    }
    
    def _extract_auxiliary_variables(self, tree: ast.AST):
        """Extract auxiliary variables from method implementations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Look for variable assignments that compute intermediate values
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                var_name = target.id
                                # Skip if already categorized or is a standard variable
                                if (var_name not in self.variables and 
                                    not var_name.startswith('_') and
                                    var_name not in ['self', 'sim_context', 't', 'stocks']):
                                    
                                    # Try to determine if it's auxiliary
                                    if self._is_auxiliary_variable(var_name, stmt):
                                        expression = self._get_expression_string(stmt.value)
                                        self.auxiliary[var_name] = {
                                            'name': var_name,
                                            'method': node.name,
                                            'line_number': stmt.lineno,
                                            'expression': expression,
                                            'python_expression': expression,
                                            'expression_source': {
                                                'start_line': getattr(stmt.value, 'lineno', None),
                                                'end_line': getattr(stmt.value, 'end_lineno', getattr(stmt.value, 'lineno', None))
                                            },
                                            'function_source': {
                                                'function_name': node.name,
                                                'start_line': getattr(node, 'lineno', None),
                                                'end_line': getattr(node, 'end_lineno', getattr(node, 'lineno', None))
                                            }
                                        }
    
    def _is_auxiliary_variable(self, var_name: str, assign_node: ast.Assign) -> bool:
        """Determine if a variable assignment represents an auxiliary variable."""
        # Skip common computed variables that shouldn't be parameters
        skip_patterns = [
            'forecast_error_factor',  # These are duplicates from parameters
            'forecasted_demand',      # Computed in methods
            'target_inventory',       # Computed in methods
            'order_quantity',         # Computed values
            'total_vars',            # Temporary variables
            'temp_',                 # Temporary variables
            'result_',               # Result variables
            'combined_',             # Combined results
        ]
        
        # Skip if it matches any skip pattern
        if any(pattern in var_name.lower() for pattern in skip_patterns):
            return False
        
        # Look for genuine auxiliary variable patterns
        auxiliary_patterns = [
            'calculated_', 'computed_', 'effective_', 'adjusted_', 
            'net_', 'average_', 'total_demand', 'capacity_'
        ]
        
        return any(pattern in var_name.lower() for pattern in auxiliary_patterns)
    
    def _get_expression_string(self, expr_node: ast.AST) -> str:
        """Get string representation of an expression."""
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(expr_node)
            except:
                pass
        return f"<expression at line {expr_node.lineno if hasattr(expr_node, 'lineno') else '?'}>"

    def _extract_flow_expressions(self, tree: ast.AST):
        """Extract symbolic expressions for flow variables from method bodies."""
        if not self.flows:
            return
        flow_ids = set(self.flows.keys())
        if not flow_ids:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assignment_map = self._collect_assignments_in_function(node)
                self._extract_flow_dict_expressions(node, assignment_map, flow_ids)
                self._extract_flow_subscript_expressions(node, assignment_map, flow_ids)

    def _extract_stock_expressions(self, tree: ast.AST):
        """Extract symbolic derivative equations for stock variables from method bodies."""
        if not self.stocks:
            return

        stock_ids = set(self.stocks.keys())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != 'calculate_derivatives_and_flows':
                continue

            assignments = self._collect_assignments_in_function(node)
            stock_derivatives = self._collect_stock_derivative_targets(node, stock_ids)
            for stock_id, derivative_node in stock_derivatives.items():
                expression, expr_start_line, expr_end_line = self._resolve_flow_expression_with_source(derivative_node, assignments)
                if not expression:
                    continue

                derivative_name = derivative_node.id if isinstance(derivative_node, ast.Name) else None
                stock_expression = f"{derivative_name} = {expression}" if derivative_name else expression

                if not self._should_update_stock_expression(stock_id, stock_expression, expr_start_line, expr_end_line):
                    continue

                self.stock_expressions[stock_id] = stock_expression
                self.stock_expression_sources[stock_id] = {
                    'start_line': expr_start_line,
                    'end_line': expr_end_line
                }
                self.stock_function_sources[stock_id] = {
                    'function_name': node.name,
                    'start_line': getattr(node, 'lineno', None),
                    'end_line': getattr(node, 'end_lineno', getattr(node, 'lineno', None))
                }
                if isinstance(self.stocks.get(stock_id), dict):
                    self.stocks[stock_id]['python_expression'] = stock_expression

    def _extract_auxiliary_expressions(self, tree: ast.AST):
        """Extract symbolic expressions for published auxiliary IDs.

        Supports both:
        - sim_context.all_params['module.var'] = expr
        - auxiliaries['auxiliary.var'] = expr
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            assignments = self._collect_assignments_in_function(node)
            append_expressions = self._collect_append_expressions_in_function(node, assignments)
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Assign):
                    continue

                for target in inner.targets:
                    aux_id = self._extract_auxiliary_id_from_target(target)
                    if not aux_id:
                        continue

                    expression, expr_start_line, expr_end_line = self._resolve_auxiliary_expression_with_source(
                        inner.value,
                        assignments,
                        append_expressions,
                    )
                    if not expression:
                        continue

                    if not self._should_update_auxiliary_expression(aux_id, expression, expr_start_line, expr_end_line):
                        continue

                    self.auxiliary_expressions[aux_id] = expression
                    self.auxiliary_expression_sources[aux_id] = {
                        'start_line': expr_start_line,
                        'end_line': expr_end_line
                    }
                    self.auxiliary_function_sources[aux_id] = {
                        'function_name': node.name,
                        'start_line': getattr(node, 'lineno', None),
                        'end_line': getattr(node, 'end_lineno', getattr(node, 'lineno', None))
                    }

                    if isinstance(self.variables.get(aux_id), dict):
                        self.variables[aux_id]['python_expression'] = expression

                    existing_aux = self.auxiliary.get(aux_id) if isinstance(self.auxiliary.get(aux_id), dict) else {}
                    variable_def = self.variables.get(aux_id) if isinstance(self.variables.get(aux_id), dict) else {}
                    auxiliary_entry = {
                        **variable_def,
                        **existing_aux,
                        'name': existing_aux.get('name') or variable_def.get('name') or aux_id,
                        'category': 'auxiliary',
                        'expression': expression,
                        'python_expression': expression,
                        'expression_source': {
                            'start_line': expr_start_line,
                            'end_line': expr_end_line,
                        },
                        'function_source': {
                            'function_name': node.name,
                            'start_line': getattr(node, 'lineno', None),
                            'end_line': getattr(node, 'end_lineno', getattr(node, 'lineno', None)),
                        },
                    }
                    if 'default' not in auxiliary_entry and 'defaultValue' not in auxiliary_entry:
                        auxiliary_entry['default'] = 0.0
                    self.auxiliary[aux_id] = auxiliary_entry

    def _extract_auxiliary_id_from_target(self, node: ast.AST) -> Optional[str]:
        """Return auxiliary id from recognized publish targets.

        Supports:
        - sim_context.all_params['module.var']
        - auxiliaries['auxiliary.var']
        """
        if not isinstance(node, ast.Subscript):
            return None

        slice_node = node.slice
        if isinstance(slice_node, ast.Index):
            slice_node = slice_node.value

        literal = self._get_literal_value(slice_node)
        if not isinstance(literal, str):
            return None

        # Pattern: sim_context.all_params['module.var']
        if isinstance(node.value, ast.Attribute):
            if (
                node.value.attr == 'all_params'
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'sim_context'
            ):
                return literal

            # Pattern: self.auxiliaries['auxiliary.var']
            if (
                node.value.attr in {'auxiliaries', 'auxiliary'}
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'self'
            ):
                return literal

        # Pattern: auxiliaries['auxiliary.var']
        if isinstance(node.value, ast.Name) and node.value.id in {'auxiliaries', 'auxiliary'}:
            return literal

        return None

    def _extract_auxiliary_id_from_sim_context_subscript(self, node: ast.AST) -> Optional[str]:
        """Backward-compatible wrapper for legacy helper name."""
        return self._extract_auxiliary_id_from_target(node)

    def _collect_append_expressions_in_function(self,
                                               func_node: ast.FunctionDef,
                                               assignments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Collect representative expressions from `<list_var>.append(expr)` calls."""
        append_expressions: Dict[str, Dict[str, Any]] = {}
        for inner in ast.walk(func_node):
            if not isinstance(inner, ast.Call):
                continue
            if not isinstance(inner.func, ast.Attribute) or inner.func.attr != 'append':
                continue
            if not isinstance(inner.func.value, ast.Name):
                continue
            if len(inner.args) != 1:
                continue

            list_name = inner.func.value.id
            expression, expr_start_line, expr_end_line = self._resolve_flow_expression_with_source(inner.args[0], assignments)
            if not expression:
                continue

            existing = append_expressions.get(list_name)
            if self._should_prefer_auxiliary_candidate(existing, expression):
                append_expressions[list_name] = {
                    'expression': expression,
                    'start_line': expr_start_line,
                    'end_line': expr_end_line,
                }
        return append_expressions

    def _is_uninformative_aux_expression(self, expression: Optional[str]) -> bool:
        """Return True when expression is a placeholder/container initializer."""
        if not expression:
            return True
        normalized = expression.strip()
        return normalized in {'[]', '{}', '()', 'list()', 'dict()', 'set()'}

    def _should_prefer_auxiliary_candidate(self,
                                           existing: Optional[Dict[str, Any]],
                                           candidate_expression: Optional[str]) -> bool:
        """Select the richest available candidate expression for auxiliary append traces."""
        if existing is None:
            return True

        existing_expression = existing.get('expression') if isinstance(existing, dict) else None
        existing_uninformative = self._is_uninformative_aux_expression(existing_expression)
        candidate_uninformative = self._is_uninformative_aux_expression(candidate_expression)

        if existing_uninformative and not candidate_uninformative:
            return True
        if not existing_uninformative and candidate_uninformative:
            return False

        existing_simple = self._is_simple_identifier_expression(existing_expression)
        candidate_simple = self._is_simple_identifier_expression(candidate_expression)
        if existing_simple and not candidate_simple:
            return True
        if not existing_simple and candidate_simple:
            return False

        return False

    def _resolve_auxiliary_expression_with_source(self,
                                                  value_node: ast.AST,
                                                  assignments: Dict[str, Dict[str, Any]],
                                                  append_expressions: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Resolve the best expression/source for an auxiliary publish assignment."""
        expression, expr_start_line, expr_end_line = self._resolve_flow_expression_with_source(value_node, assignments)

        if isinstance(value_node, ast.Name):
            list_name = value_node.id
            append_candidate = append_expressions.get(list_name)
            if append_candidate and self._is_uninformative_aux_expression(expression):
                return (
                    append_candidate.get('expression'),
                    append_candidate.get('start_line'),
                    append_candidate.get('end_line'),
                )

        return expression, expr_start_line, expr_end_line

    def _should_update_auxiliary_expression(self,
                                            aux_id: str,
                                            candidate_expression: Optional[str],
                                            candidate_start_line: Optional[int],
                                            candidate_end_line: Optional[int]) -> bool:
        """Decide whether candidate auxiliary expression metadata should replace existing data."""
        existing_expression = self.auxiliary_expressions.get(aux_id)
        existing_source = self.auxiliary_expression_sources.get(aux_id) or {}
        existing_start_line = existing_source.get('start_line')

        if not existing_expression:
            return True

        existing_is_simple = self._is_simple_identifier_expression(existing_expression)
        candidate_is_simple = self._is_simple_identifier_expression(candidate_expression)

        if existing_is_simple and not candidate_is_simple:
            return True
        if not existing_is_simple and candidate_is_simple:
            return False

        if existing_start_line is None and candidate_start_line is not None:
            return True
        if existing_start_line is not None and candidate_start_line is None:
            return False

        if (candidate_expression or '').strip() == (existing_expression or '').strip():
            if candidate_start_line is not None and existing_start_line is not None:
                return candidate_start_line < existing_start_line

        return False

    def _collect_stock_derivative_targets(self,
                                          func_node: ast.FunctionDef,
                                          stock_ids: Set[str]) -> Dict[str, ast.AST]:
        """Collect stock->derivative mappings from `derivatives` containers."""
        stock_derivatives: Dict[str, ast.AST] = {}
        for inner in ast.walk(func_node):
            if not isinstance(inner, ast.Assign):
                continue

            for target in inner.targets:
                # Pattern: derivatives = {'module.s_Stock': d_Stock_dt}
                if isinstance(target, ast.Name) and target.id == 'derivatives' and isinstance(inner.value, ast.Dict):
                    for key_node, value_node in zip(inner.value.keys, inner.value.values):
                        stock_id = self._get_literal_value(key_node)
                        if isinstance(stock_id, str) and stock_id in stock_ids:
                            stock_derivatives[stock_id] = value_node

                # Pattern: derivatives['module.s_Stock'] = d_Stock_dt
                stock_id = self._extract_derivative_stock_id_from_subscript(target)
                if stock_id and stock_id in stock_ids:
                    stock_derivatives[stock_id] = inner.value

        return stock_derivatives

    def _extract_derivative_stock_id_from_subscript(self, node: ast.AST) -> Optional[str]:
        """Return stock id from AST target `derivatives['id']` if present."""
        if not isinstance(node, ast.Subscript):
            return None

        if not (isinstance(node.value, ast.Name) and node.value.id == 'derivatives'):
            return None

        slice_node = node.slice
        if isinstance(slice_node, ast.Index):
            slice_node = slice_node.value

        literal = self._get_literal_value(slice_node)
        return literal if isinstance(literal, str) else None

    def _extract_flow_subscript_expressions(self,
                                           func_node: ast.FunctionDef,
                                           assignments: Dict[str, Dict[str, Any]],
                                           flow_ids: Set[str]):
        """Link flow IDs for assignments like flows['var.id'] = expr."""
        for inner in ast.walk(func_node):
            if not isinstance(inner, ast.Assign):
                continue

            expression, expr_start_line, expr_end_line = self._resolve_flow_expression_with_source(inner.value, assignments)
            if not expression:
                continue

            for target in inner.targets:
                flow_id = self._extract_flow_id_from_subscript(target)
                if not flow_id or flow_id not in flow_ids:
                    continue

                if not self._should_update_flow_expression(flow_id, expression, expr_start_line, expr_end_line):
                    continue

                self.flow_expressions[flow_id] = expression
                self.flow_expression_sources[flow_id] = {
                    'start_line': expr_start_line,
                    'end_line': expr_end_line
                }
                self.flow_function_sources[flow_id] = {
                    'function_name': func_node.name,
                    'start_line': getattr(func_node, 'lineno', None),
                    'end_line': getattr(func_node, 'end_lineno', getattr(func_node, 'lineno', None))
                }

                if isinstance(self.flows.get(flow_id), dict):
                    self.flows[flow_id]['python_expression'] = expression

    def _extract_flow_id_from_subscript(self, node: ast.AST) -> Optional[str]:
        """Return flow id from AST target `flows['id']` if present."""
        if not isinstance(node, ast.Subscript):
            return None

        if not (isinstance(node.value, ast.Name) and node.value.id == 'flows'):
            return None

        slice_node = node.slice
        # Python <3.9 compatibility where slice is wrapped
        if isinstance(slice_node, ast.Index):
            slice_node = slice_node.value

        literal = self._get_literal_value(slice_node)
        return literal if isinstance(literal, str) else None

    def _collect_assignments_in_function(self, func_node: ast.FunctionDef) -> Dict[str, Dict[str, Any]]:
        """Collect all simple assignments within a function for later lookup.
        
        VMS Single-Assignment Convention: Each variable should be assigned exactly once.
        If a variable is assigned multiple times, a warning is recorded. For conditional
        logic, use helper functions instead of if/else blocks with multiple assignments.
        
        Also extracts dependencies from each assignment's right-hand side.
        """
        assignments: Dict[str, Dict[str, Any]] = {}
        assignment_counts: Dict[str, int] = {}  # Track how many times each var is assigned
        
        for inner in ast.walk(func_node):
            if isinstance(inner, ast.Assign):
                value_expr = self._get_expression_string(inner.value)
                dependencies = self._extract_expression_dependencies(inner.value)
                
                for target in inner.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        
                        # Track assignment count
                        assignment_counts[var_name] = assignment_counts.get(var_name, 0) + 1
                        
                        # Record duplicate assignment warning
                        if assignment_counts[var_name] > 1:
                            self.duplicate_assignments.append({
                                'variable': var_name,
                                'function': func_node.name,
                                'line': inner.lineno,
                                'count': assignment_counts[var_name],
                                'message': f"VMS Convention Warning: '{var_name}' assigned multiple times in {func_node.name}(). Use helper function for conditional logic."
                            })
                        
                        assignments[var_name] = {
                            'expression': value_expr,
                            'start_line': getattr(inner.value, 'lineno', None),
                            'end_line': getattr(inner.value, 'end_lineno', getattr(inner.value, 'lineno', None))
                        }
                        
                        # Store dependencies for this variable
                        if dependencies:
                            self.variable_dependencies[var_name] = dependencies
                            
            elif isinstance(inner, ast.AnnAssign):
                if isinstance(inner.target, ast.Name) and inner.value is not None:
                    var_name = inner.target.id
                    assignments[var_name] = {
                        'expression': self._get_expression_string(inner.value),
                        'start_line': getattr(inner.value, 'lineno', None),
                        'end_line': getattr(inner.value, 'end_lineno', getattr(inner.value, 'lineno', None))
                    }
                    dependencies = self._extract_expression_dependencies(inner.value)
                    if dependencies:
                        self.variable_dependencies[var_name] = dependencies
                        
        return assignments
    
    def _extract_expression_dependencies(self, expr_node: ast.AST) -> List[str]:
        """Extract variable dependencies from an expression (shallow extraction).
        
        For function calls like calc_flow(rate, factor), extracts 'rate' and 'factor'
        as dependencies without tracing into the function body.
        
        Returns:
            List of variable names that the expression depends on
        """
        dependencies = []
        
        for node in ast.walk(expr_node):
            # Direct variable references
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                var_name = node.id
                # Skip built-ins and common names
                if var_name not in ['self', 'sim_context', 'True', 'False', 'None', 
                                     'min', 'max', 'abs', 'sum', 'len', 'range', 'float', 'int']:
                    if var_name not in dependencies:
                        dependencies.append(var_name)
            
            # Function call arguments - shallow extraction
            elif isinstance(node, ast.Call):
                # Extract positional arguments
                for arg in node.args:
                    if isinstance(arg, ast.Name) and isinstance(arg.ctx, ast.Load):
                        var_name = arg.id
                        if var_name not in ['self', 'sim_context'] and var_name not in dependencies:
                            dependencies.append(var_name)
                
                # Extract keyword arguments
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Name) and isinstance(kw.value.ctx, ast.Load):
                        var_name = kw.value.id
                        if var_name not in ['self', 'sim_context'] and var_name not in dependencies:
                            dependencies.append(var_name)
                
                # Handle sim_context.get_param('name') and sim_context.get_stock('name')
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == 'sim_context' and
                    node.func.attr in ['get_param', 'get_stock'] and
                    node.args and isinstance(node.args[0], ast.Constant)):
                    param_name = node.args[0].value
                    if isinstance(param_name, str) and param_name not in dependencies:
                        dependencies.append(param_name)
        
        return dependencies

    def _extract_flow_dict_expressions(self,
                                       func_node: ast.FunctionDef,
                                       assignments: Dict[str, Dict[str, Any]],
                                       flow_ids: Set[str]):
        """Link flow IDs in dict literals to their underlying expressions."""
        for inner in ast.walk(func_node):
            if isinstance(inner, ast.Dict):
                for key_node, value_node in zip(inner.keys, inner.values):
                    flow_id = self._get_literal_value(key_node)
                    if isinstance(flow_id, str) and flow_id in flow_ids:
                        expression, expr_start_line, expr_end_line = self._resolve_flow_expression_with_source(value_node, assignments)
                        if expression:
                            if not self._should_update_flow_expression(flow_id, expression, expr_start_line, expr_end_line):
                                continue

                            self.flow_expressions[flow_id] = expression
                            self.flow_expression_sources[flow_id] = {
                                'start_line': expr_start_line,
                                'end_line': expr_end_line
                            }
                            self.flow_function_sources[flow_id] = {
                                'function_name': func_node.name,
                                'start_line': getattr(func_node, 'lineno', None),
                                'end_line': getattr(func_node, 'end_lineno', getattr(func_node, 'lineno', None))
                            }
                            if isinstance(self.flows.get(flow_id), dict):
                                self.flows[flow_id]['python_expression'] = expression

    def _is_simple_identifier_expression(self, expression: Optional[str]) -> bool:
        """Return True when expression is a bare identifier (e.g. `ship_steel`)."""
        if not expression:
            return False
        return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', expression.strip()))

    def _should_update_flow_expression(self,
                                       flow_id: str,
                                       candidate_expression: Optional[str],
                                       candidate_start_line: Optional[int],
                                       candidate_end_line: Optional[int]) -> bool:
        """Decide whether candidate flow expression metadata should replace existing data.

        This avoids regressing to alias-only rows from flow maps when an assignment-backed
        expression/source has already been discovered.
        """
        existing_expression = self.flow_expressions.get(flow_id)
        existing_source = self.flow_expression_sources.get(flow_id) or {}
        existing_start_line = existing_source.get('start_line')

        if not existing_expression:
            return True

        existing_is_simple = self._is_simple_identifier_expression(existing_expression)
        candidate_is_simple = self._is_simple_identifier_expression(candidate_expression)

        # Prefer richer expressions over alias-only identifiers.
        if existing_is_simple and not candidate_is_simple:
            return True
        if not existing_is_simple and candidate_is_simple:
            return False

        # Prefer candidates with concrete source lines when existing source is missing.
        if existing_start_line is None and candidate_start_line is not None:
            return True
        if existing_start_line is not None and candidate_start_line is None:
            return False

        # If both expressions are equivalent, keep the earliest source row (assignment).
        if (candidate_expression or '').strip() == (existing_expression or '').strip():
            if candidate_start_line is not None and existing_start_line is not None:
                return candidate_start_line < existing_start_line

        # Otherwise keep the first non-empty expression to avoid noisy oscillation.
        return False

    def _should_update_stock_expression(self,
                                        stock_id: str,
                                        candidate_expression: Optional[str],
                                        candidate_start_line: Optional[int],
                                        candidate_end_line: Optional[int]) -> bool:
        """Decide whether candidate stock expression metadata should replace existing data."""
        existing_expression = self.stock_expressions.get(stock_id)
        existing_source = self.stock_expression_sources.get(stock_id) or {}
        existing_start_line = existing_source.get('start_line')

        if not existing_expression:
            return True

        existing_is_simple = self._is_simple_identifier_expression(existing_expression)
        candidate_is_simple = self._is_simple_identifier_expression(candidate_expression)

        if existing_is_simple and not candidate_is_simple:
            return True
        if not existing_is_simple and candidate_is_simple:
            return False

        if existing_start_line is None and candidate_start_line is not None:
            return True
        if existing_start_line is not None and candidate_start_line is None:
            return False

        if (candidate_expression or '').strip() == (existing_expression or '').strip():
            if candidate_start_line is not None and existing_start_line is not None:
                return candidate_start_line < existing_start_line

        return False

    def _resolve_flow_expression(self, value_node: ast.AST, assignments: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Resolve the best available expression string for a flow value node."""
        expression, _start_line, _end_line = self._resolve_flow_expression_with_source(value_node, assignments)
        return expression

    def _resolve_flow_expression_with_source(self,
                                             value_node: ast.AST,
                                             assignments: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Resolve expression text and source span for a flow value node.

        If the flow references an intermediate variable (e.g. `ship_steel`) this
        returns the source of that variable assignment, not the later flow dict row.
        """
        if isinstance(value_node, ast.Name):
            var_name = value_node.id
            if var_name in assignments:
                assignment_info = assignments[var_name] or {}
                return (
                    assignment_info.get('expression'),
                    assignment_info.get('start_line'),
                    assignment_info.get('end_line')
                )
            return (
                var_name,
                getattr(value_node, 'lineno', None),
                getattr(value_node, 'end_lineno', getattr(value_node, 'lineno', None))
            )
        return (
            self._get_expression_string(value_node),
            getattr(value_node, 'lineno', None),
            getattr(value_node, 'end_lineno', getattr(value_node, 'lineno', None))
        )

    def _extract_calc_function_definitions(self, tree: ast.AST, source_code: str, file_path: str):
        """Capture helper function definitions for calc_* calls referenced by flow expressions."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith('calc_'):
                continue

            self.calc_function_definitions[node.name] = {
                'name': node.name,
                'file': file_path,
                'start_line': getattr(node, 'lineno', None),
                'end_line': getattr(node, 'end_lineno', getattr(node, 'lineno', None)),
                'definition': ast.get_source_segment(source_code, node)
            }


def parse_veydra_model(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to parse a single Veydra model file.
    
    Args:
        file_path: Path to Python file containing Veydra model
        
    Returns:
        Dictionary with extracted model components
    """
    parser = VeydraModelASTParser()
    return parser.parse_file(file_path)


def parse_veydra_model_directory(directory_path: str) -> Dict[str, Any]:
    """
    Parse all Python files in a directory for Veydra model components.
    
    Args:
        directory_path: Path to directory containing Veydra model files
        
    Returns:
        Dictionary with combined results from all files
    """
    directory = Path(directory_path)
    parser = VeydraModelASTParser()
    
    combined_results = {
        'files': {},
        'all_variables': {},
        'all_parameters': {},
        'all_stocks': {},
        'all_flows': {},
        'all_auxiliary': {},
        'all_flow_expressions': {},
        'all_flow_expression_sources': {},
        'all_flow_function_sources': {},
        'all_stock_expressions': {},
        'all_stock_expression_sources': {},
        'all_stock_function_sources': {},
        'all_auxiliary_expressions': {},
        'all_auxiliary_expression_sources': {},
        'all_auxiliary_function_sources': {},
        'all_calc_function_definitions': {},
        'all_submodels': [],
        'summary': {}
    }
    
    # Find all Python files
    python_files = list(directory.glob('*.py'))
    
    for py_file in python_files:
        try:
            result = parser.parse_file(str(py_file))
            file_key = py_file.name
            
            combined_results['files'][file_key] = result
            
            # Merge results
            combined_results['all_variables'].update(result['variables'])
            combined_results['all_parameters'].update(result['parameters'])
            combined_results['all_stocks'].update(result['stocks'])
            combined_results['all_flows'].update(result['flows'])
            combined_results['all_flow_expressions'].update(result.get('flow_expressions', {}))
            combined_results['all_stock_expressions'].update(result.get('stock_expressions', {}))
            combined_results['all_auxiliary_expressions'].update(result.get('auxiliary_expressions', {}))
            flow_expression_sources = result.get('flow_expression_sources', {})
            flow_function_sources = result.get('flow_function_sources', {})
            stock_expression_sources = result.get('stock_expression_sources', {})
            stock_function_sources = result.get('stock_function_sources', {})
            auxiliary_expression_sources = result.get('auxiliary_expression_sources', {})
            auxiliary_function_sources = result.get('auxiliary_function_sources', {})
            calc_function_definitions = result.get('calc_function_definitions', {})
            file_path = result.get('metadata', {}).get('file_path', str(py_file))

            for flow_id, source_info in flow_expression_sources.items():
                combined_results['all_flow_expression_sources'][flow_id] = {
                    'file': file_path,
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for flow_id, source_info in flow_function_sources.items():
                combined_results['all_flow_function_sources'][flow_id] = {
                    'file': file_path,
                    'function_name': source_info.get('function_name'),
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for stock_id, source_info in stock_expression_sources.items():
                combined_results['all_stock_expression_sources'][stock_id] = {
                    'file': file_path,
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for stock_id, source_info in stock_function_sources.items():
                combined_results['all_stock_function_sources'][stock_id] = {
                    'file': file_path,
                    'function_name': source_info.get('function_name'),
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for aux_id, source_info in auxiliary_expression_sources.items():
                combined_results['all_auxiliary_expression_sources'][aux_id] = {
                    'file': file_path,
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for aux_id, source_info in auxiliary_function_sources.items():
                combined_results['all_auxiliary_function_sources'][aux_id] = {
                    'file': file_path,
                    'function_name': source_info.get('function_name'),
                    'start_line': source_info.get('start_line'),
                    'end_line': source_info.get('end_line')
                }

            for function_name, function_info in calc_function_definitions.items():
                combined_results['all_calc_function_definitions'][function_name] = {
                    'name': function_name,
                    'file': file_path,
                    'start_line': function_info.get('start_line'),
                    'end_line': function_info.get('end_line'),
                    'definition': function_info.get('definition')
                }
            for aux_id, aux_info in result.get('auxiliary', {}).items():
                if isinstance(aux_info, dict):
                    aux_payload = dict(aux_info)
                else:
                    aux_payload = {
                        'name': aux_id,
                        'expression': aux_info,
                        'python_expression': aux_info,
                    }

                aux_payload.setdefault('file', file_path)

                expression_source = aux_payload.get('expression_source')
                if isinstance(expression_source, dict):
                    aux_payload['expression_source'] = {
                        'file': file_path,
                        'start_line': expression_source.get('start_line'),
                        'end_line': expression_source.get('end_line')
                    }
                elif aux_payload.get('line_number') is not None:
                    aux_payload['expression_source'] = {
                        'file': file_path,
                        'start_line': aux_payload.get('line_number'),
                        'end_line': aux_payload.get('line_number')
                    }

                function_source = aux_payload.get('function_source')
                if isinstance(function_source, dict):
                    aux_payload['function_source'] = {
                        'file': file_path,
                        'function_name': function_source.get('function_name'),
                        'start_line': function_source.get('start_line'),
                        'end_line': function_source.get('end_line')
                    }

                combined_results['all_auxiliary'][aux_id] = aux_payload
            combined_results['all_submodels'].extend(result['submodels'])
            
        except Exception as e:
            combined_results['files'][py_file.name] = {
                'error': str(e),
                'variables': {},
                'parameters': {},
                'stocks': {},
                'flows': {},
                'stock_expressions': {},
                'auxiliary': {},
                'submodels': []
            }
    
    # Generate summary
    combined_results['summary'] = {
        'files_processed': len(python_files),
        'total_variables': len(combined_results['all_variables']),
        'total_parameters': len(combined_results['all_parameters']),
        'total_stocks': len(combined_results['all_stocks']),
        'total_flows': len(combined_results['all_flows']),
        'total_auxiliary': len(combined_results['all_auxiliary']),
        'total_submodels': len(combined_results['all_submodels'])
    }
    
    return combined_results


def generate_model_parameters_json(model_file_path: str, output_file: Optional[str] = None, logger=None) -> Dict[str, Any]:
    """
    Generate a clean model-parameters.json file from the analyzed model structure.
    
    Args:
        model_file_path: Path to the main model file or directory
        output_file: Optional output file path. If None, returns the structure
        logger: Optional logger for status messages
        
    Returns:
        Dictionary containing the model parameters JSON structure
    """
    if logger:
        logger.info(f"Parsing model parameters from: {model_file_path}")
    
    flow_expression_lookup: Dict[str, str] = {}
    flow_expression_sources: Dict[str, Dict[str, Any]] = {}
    flow_function_sources: Dict[str, Dict[str, Any]] = {}
    calc_function_definitions: Dict[str, Dict[str, Any]] = {}

    # If it's a directory, analyze all files, otherwise just the single file
    if Path(model_file_path).is_dir():
        all_results = parse_veydra_model_directory(model_file_path)
        all_variables = all_results['all_variables']
        all_parameters = all_results['all_parameters']
        all_stocks = all_results['all_stocks']
        all_flows = all_results['all_flows']
        all_auxiliary = all_results['all_auxiliary']
        flow_expression_lookup = {
            **all_results.get('all_flow_expressions', {}),
            **all_results.get('all_stock_expressions', {}),
            **all_results.get('all_auxiliary_expressions', {}),
        }
        flow_expression_sources = {
            **all_results.get('all_flow_expression_sources', {}),
            **all_results.get('all_stock_expression_sources', {}),
            **all_results.get('all_auxiliary_expression_sources', {}),
        }
        flow_function_sources = {
            **all_results.get('all_flow_function_sources', {}),
            **all_results.get('all_stock_function_sources', {}),
            **all_results.get('all_auxiliary_function_sources', {}),
        }
        calc_function_definitions = all_results.get('all_calc_function_definitions', {})
        # Combine auxiliary from all files for function string extraction
        combined_auxiliary = {}
        for file_result in all_results['files'].values():
            if not isinstance(file_result, dict) or 'auxiliary' not in file_result:
                continue
            combined_auxiliary.update(file_result.get('auxiliary', {}))
        file_analysis = {'auxiliary': combined_auxiliary}
    else:
        result = parse_veydra_model(model_file_path)
        all_variables = result['variables']
        all_parameters = result['parameters']
        all_stocks = result['stocks']
        all_flows = result['flows']
        all_auxiliary = result['auxiliary']
        flow_expression_lookup = {
            **result.get('flow_expressions', {}),
            **result.get('stock_expressions', {}),
            **result.get('auxiliary_expressions', {}),
        }
        flow_expression_sources = {
            **result.get('flow_expression_sources', {}),
            **result.get('stock_expression_sources', {}),
            **result.get('auxiliary_expression_sources', {}),
        }
        flow_function_sources = {
            **result.get('flow_function_sources', {}),
            **result.get('stock_function_sources', {}),
            **result.get('auxiliary_function_sources', {}),
        }
        calc_function_definitions = result.get('calc_function_definitions', {})
        file_analysis = result

    def _normalize_source_path(file_path_value: Optional[str]) -> Optional[str]:
        if not file_path_value:
            return None
        try:
            source_path = Path(file_path_value).resolve()
            model_base = Path(model_file_path).resolve()
            return str(source_path.relative_to(model_base)).replace('\\', '/')
        except Exception:
            return str(file_path_value).replace('\\', '/')

    def _extract_calc_function_calls(expression: str) -> List[str]:
        if not expression:
            return []
        seen: Set[str] = set()
        calls: List[str] = []
        for match in re.finditer(r'\b(calc_[A-Za-z0-9_]+)\s*\(', expression):
            function_name = match.group(1)
            if function_name in seen:
                continue
            seen.add(function_name)
            calls.append(function_name)
        return calls
    
    parameters_list = []
    
    # Helper function to extract namespace/group from variable name
    def extract_namespace_and_group(var_name: str) -> Tuple[str, str]:
        """Extract namespace and group from variable name."""
        parts = var_name.split('.')
        if len(parts) >= 2:
            namespace = parts[0]
            # Create a readable group name
            if namespace == 'simulation':
                return namespace, 'Simulation Control'
            elif namespace == 'customer_demand':
                return namespace, 'Customer Demand'
            elif namespace == 'retailer':
                return namespace, 'Retailer Policies'
            elif namespace == 'distributor':
                return namespace, 'Distributor Policies'
            elif namespace == 'manufacturer':
                return namespace, 'Manufacturer Policies'
            else:
                # Capitalize first letter and replace underscores
                readable_name = namespace.replace('_', ' ').title()
                return namespace, readable_name
        else:
            # Handle variables without namespace
            return 'general', 'General Parameters'
    
    # Process all variables
    all_vars_to_process = [
        (all_parameters, 'parameter'),
        (all_stocks, 'stock'),
        (all_flows, 'flow'),
        (all_auxiliary, 'auxiliary')
    ]
    
    for var_dict, category in all_vars_to_process:
        for var_name, var_def in var_dict.items():
            if not isinstance(var_def, dict):
                continue

            # Skip local helper aliases when a namespaced variable already exists.
            # Example: `effective_unit_cost` should defer to `market.effective_unit_cost`.
            if category == 'auxiliary' and isinstance(var_name, str) and '.' not in var_name:
                has_namespaced_alias = any(
                    isinstance(candidate_name, str) and candidate_name.endswith(f".{var_name}")
                    for candidate_name in all_variables.keys()
                )
                if has_namespaced_alias:
                    continue
            
            namespace, group_name = extract_namespace_and_group(var_name)
            
            # Determine parameter type
            if var_name.startswith('simulation.'):
                param_type = 'simulation_setting'
            elif category == 'stock':
                param_type = 'initial_value'
            elif category == 'rate' or category == 'flow':
                param_type = 'flow_parameter'
            elif category == 'parameter':
                param_type = 'parameter'
            else:
                param_type = 'auxiliary_initial'
            
            # Parse units into components
            units_str = var_def.get('units', 'dimensionless')
            units_numerator, units_denominator = _parse_units_components(units_str)
            
            # Create parameter entry
            param_entry = {
                "id": var_name,
                "python_variable_id": var_name,
                "label": var_def.get('name', var_name.replace('_', ' ').title()),
                "min": var_def.get('min', 0.0),
                "max": var_def.get('max', 100.0),
                "step": var_def.get('step', 1.0),
                "defaultValue": var_def.get('default', 0.0),
                "group": group_name,
                "namespace": namespace,
                "help": var_def.get('description', f"Parameter for {var_name}"),
                "units": units_str,
                "units_numerator": units_numerator,
                "units_denominator": units_denominator,
                "type": param_type,
                "category": category
            }

            formula_definition = flow_expression_lookup.get(var_name)
            if not formula_definition:
                python_expression = var_def.get('python_expression')
                if isinstance(python_expression, str) and python_expression.strip():
                    formula_definition = python_expression.strip()
            if formula_definition:
                param_entry['expression_definition'] = formula_definition
                param_entry['formula_definition'] = formula_definition

                calc_calls = _extract_calc_function_calls(formula_definition)
                if calc_calls:
                    param_entry['expression_function_calls'] = calc_calls
                    primary_calc = calc_calls[0]
                    calc_info = calc_function_definitions.get(primary_calc)
                    if isinstance(calc_info, dict):
                        param_entry['python_function_definition_name'] = primary_calc
                        param_entry['python_function_definition'] = calc_info.get('definition')
                        normalized_calc_file = _normalize_source_path(calc_info.get('file'))
                        if normalized_calc_file:
                            param_entry['python_function_definition_source'] = {
                                'file': normalized_calc_file,
                                'start_line': calc_info.get('start_line'),
                                'end_line': calc_info.get('end_line')
                            }

            formula_source = flow_expression_sources.get(var_name)
            if not isinstance(formula_source, dict):
                candidate_expression_source = var_def.get('expression_source')
                if isinstance(candidate_expression_source, dict):
                    formula_source = candidate_expression_source
            if isinstance(formula_source, dict):
                normalized_formula_file = _normalize_source_path(formula_source.get('file'))
                if normalized_formula_file:
                    param_entry['expression_source'] = {
                        'file': normalized_formula_file,
                        'start_line': formula_source.get('start_line'),
                        'end_line': formula_source.get('end_line')
                    }
                    param_entry['formula_source'] = {
                        'file': normalized_formula_file,
                        'start_line': formula_source.get('start_line'),
                        'end_line': formula_source.get('end_line')
                    }

            function_source = flow_function_sources.get(var_name)
            if not isinstance(function_source, dict):
                candidate_function_source = var_def.get('function_source')
                if isinstance(candidate_function_source, dict):
                    function_source = candidate_function_source
            if isinstance(function_source, dict):
                normalized_function_file = _normalize_source_path(function_source.get('file'))
                if normalized_function_file:
                    param_entry['function_source'] = {
                        'file': normalized_function_file,
                        'start_line': function_source.get('start_line'),
                        'end_line': function_source.get('end_line')
                    }
            
            # For flow and auxiliary variables, adjust constraints if not specified
            if category in ['flow', 'auxiliary'] and var_def.get('min') is None:
                # Set reasonable defaults for flows/auxiliaries
                if 'rate' in var_name or '_dt' in var_name:
                    param_entry['min'] = -1000.0
                    param_entry['max'] = 1000.0
                    param_entry['step'] = 1.0
            
            # For stocks, ensure we have reasonable initial value constraints
            if category == 'stock':
                if var_def.get('min') is None:
                    param_entry['min'] = 0.0
                if var_def.get('max') is None:
                    param_entry['max'] = var_def.get('default', 1000.0) * 5  # 5x default as max
            
            parameters_list.append(param_entry)
    
    # Sort parameters by group and then by name for better organization
    parameters_list.sort(key=lambda x: (x['group'], x['label']))
    
    # Create the final structure
    model_parameters = {
        "_comment": "Auto-generated model parameters from Veydra AST analysis",
        "generation_info": {
            "generated_by": "veydra_ast_parser.py",
            "source_file": str(model_file_path),
            "generated_at": datetime.now().isoformat(),
            "total_parameters": len(parameters_list)
        },
        "parameters": parameters_list,
        "metadata": {
            "name": f"Model Parameters - {Path(model_file_path).stem}",
            "description": "Parameter configuration extracted from model source code",
            "parameter_types": {
                "parameter": "User-adjustable parameters",
                "initial_value": "Initial stock values",
                "flow_parameter": "Flow rate parameters",
                "auxiliary_initial": "Initial auxiliary values",
                "simulation_setting": "Simulation control settings"
            }
        }
    }
    
    # Write to file if output_file is specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_parameters, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"✅ Generated model-parameters.json: {output_file}")
            logger.info(f"   Total parameters: {len(parameters_list)}")
        
        # Show summary by group
        groups = {}
        for param in parameters_list:
            group = param['group']
            if group not in groups:
                groups[group] = []
            groups[group].append(param)
        
        if logger:
            logger.info(f"   Groups:")
            for group, params in groups.items():
                logger.info(f"     • {group}: {len(params)} parameters")
    
    return model_parameters


def _parse_units_components(units_string: str) -> Tuple[str, str]:
    """
    Parse units string into numerator and denominator components.
    
    Args:
        units_string: Units string like "units per day", "dimensionless", "units * kg per day * second"
        
    Returns:
        Tuple of (numerator, denominator) where:
        - numerator: units in numerator (e.g., "units", "units * kg")
        - denominator: units in denominator (e.g., "day", "day * second", "1" for non-rates)
    """
    if not units_string or units_string.lower().strip() in ['dimensionless', 'none', '']:
        return 'dimensionless', '1'
    
    # Clean the string
    units_clean = units_string.strip().lower()
    
    # Handle special formats
    if 'yyyy-mm-dd' in units_clean:
        return 'date', '1'
    
    # Handle "per" cases - split on the first "per" only
    if ' per ' in units_clean:
        parts = units_clean.split(' per ', 1)
        numerator = parts[0].strip()
        denominator = parts[1].strip()
        
        # Handle multiple "per" in denominator by treating as multiplication
        denominator = denominator.replace(' per ', ' * ')
    elif '/unit' in units_clean or '/day' in units_clean or '/second' in units_clean:
        # Handle slash notation
        if '/' in units_clean:
            parts = units_clean.split('/', 1)
            numerator = parts[0].strip()
            denominator = parts[1].strip()
        else:
            numerator = units_clean
            denominator = '1'
    else:
        # No rate, just units
        numerator = units_clean
        denominator = '1'
    
    # Clean up common abbreviations and normalize
    def normalize_unit(unit_str):
        if not unit_str or unit_str == '1':
            return unit_str
            
        unit_str = unit_str.strip()
        
        # Replace common abbreviations
        replacements = {
            'units': 'unit',
            'days': 'day', 
            'seconds': 'second',
            'minutes': 'minute',
            'hours': 'hour',
            'weeks': 'week',
            'months': 'month',
            'years': 'year'
        }
        
        # Split by both spaces and * to handle compound units
        # First, normalize * with spaces around them
        unit_str = unit_str.replace('*', ' * ')
        
        # Split and process each token
        tokens = []
        for token in unit_str.split():
            token = token.strip()
            if token == '*':
                tokens.append('*')
            elif token:
                # Apply replacements
                tokens.append(replacements.get(token, token))
        
        # Remove duplicate * and clean up
        clean_tokens = []
        for i, token in enumerate(tokens):
            if token == '*':
                # Only add * if it's between two actual units and not already added
                if (i > 0 and i < len(tokens) - 1 and 
                    tokens[i-1] != '*' and tokens[i+1] != '*' and
                    (not clean_tokens or clean_tokens[-1] != '*')):
                    clean_tokens.append('*')
            elif token:
                clean_tokens.append(token)
        
        return ' '.join(clean_tokens) if clean_tokens else unit_str
    
    numerator = normalize_unit(numerator) if numerator else 'dimensionless'
    denominator = normalize_unit(denominator) if denominator != '1' else '1'
    
    return numerator, denominator


def _extract_python_function_string(var_name: str, var_def: Dict[str, Any], file_analysis: Dict[str, Any]) -> str:
    """
    Extract or construct a Python function string for a variable.
    
    This attempts to find how the variable is calculated or used in the model code.
    """
    # For parameters, the function is usually just getting the parameter value
    if var_def.get('category') == 'parameter':
        return f"sim_context.get_param('{var_name}', {var_def.get('default', 0.0)})"
    
    # For stocks, it's getting the stock value
    elif var_def.get('category') == 'stock':
        return f"sim_context.get_stock('{var_name}', {var_def.get('default', 0.0)})"
    
    # For flows, try to find the calculation in auxiliary variables or return a placeholder
    elif var_def.get('category') in ['flow', 'rate']:
        # Look for this variable in auxiliary calculations
        for aux_name, aux_info in file_analysis.get('auxiliary', {}).items():
            if var_name.split('.')[-1] in aux_name:
                expression = aux_info.get('expression', '')
                if expression and expression != f"<expression at line {aux_info.get('line_number', '?')}>":
                    return expression
        
        # Default calculation for flows
        return f"calculated_{var_name.replace('.', '_')}(sim_context)"
    
    # For auxiliary variables, try to extract the actual expression
    elif var_def.get('category') == 'auxiliary':
        if var_name in file_analysis.get('auxiliary', {}):
            aux_info = file_analysis['auxiliary'][var_name]
            expression = aux_info.get('expression', '')
            if expression and expression != f"<expression at line {aux_info.get('line_number', '?')}>":
                return expression
        
        return f"calculate_{var_name.replace('.', '_')}(sim_context)"
    
    # Default case
    return f"get_value('{var_name}')"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_or_dir = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        generate_model_parameters_json(file_or_dir, output_file)
    else:
        print("Usage: python veydra_ast_parser.py <file_or_directory> [output_file]")