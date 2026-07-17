"""
Feedback loop detection module for causal loop diagrams.

This module provides functions to detect feedback loops in causal relationships
using depth-first search algorithms.
"""

import textwrap
import os

def truncate_name(name, max_length=25):
    """Truncate long variable names for display purposes"""
    if len(name) <= max_length:
        return name
    return name[:max_length-3] + "..."

def format_loop_path(variables, max_width=80):
    """Format a loop path with proper line wrapping"""
    if not variables:
        return ""
    
    # Create the full path
    path_elements = variables + [variables[0]]  # Close the loop
    
    # Try to fit on one line first
    full_path = " → ".join(path_elements)
    if len(full_path) <= max_width:
        return full_path
    
    # If too long, use truncated names
    truncated_elements = [truncate_name(var) for var in path_elements]
    truncated_path = " → ".join(truncated_elements)
    if len(truncated_path) <= max_width:
        return truncated_path
    
    # If still too long, split across multiple lines
    lines = []
    current_line = []
    current_length = 0
    
    for i, element in enumerate(truncated_elements):
        element_with_arrow = element + (" → " if i < len(truncated_elements) - 1 else "")
        
        if current_length + len(element_with_arrow) > max_width and current_line:
            lines.append("".join(current_line))
            current_line = ["    " + element_with_arrow]  # Indent continuation
            current_length = len(element_with_arrow) + 4
        else:
            current_line.append(element_with_arrow)
            current_length += len(element_with_arrow)
    
    if current_line:
        lines.append("".join(current_line))
    
    return "\n".join(lines)

def get_terminal_width():
    """Get terminal width, default to 80 if not available"""
    try:
        return os.get_terminal_size().columns
    except:
        return 80

def find_feedback_loops(causal_links, max_loop_length=50):
    """
    Detect feedback loops in causal relationships using depth-first search.
    
    Args:
        causal_links: List of dicts with 'source', 'target', 'polarity'
        max_loop_length: Maximum length of loops to detect (prevents infinite search)
    
    Returns:
        dict: Contains loops data, statistics, and CLD-ready structure
    """
    # Build adjacency list representation
    graph = {}
    edges = {}  # Store edge details for loop analysis
    
    for link in causal_links:
        source = link['source']
        target = link['target']
        polarity = link['polarity']
        
        if source not in graph:
            graph[source] = []
        graph[source].append(target)
        edges[(source, target)] = polarity
    
    print(f"📊 Built graph with {len(graph)} nodes and {len(causal_links)} edges")
    print(f"🔍 Nodes: {sorted(list(graph.keys())[:10])}{'...' if len(graph) > 10 else ''}")
    
    # Find all feedback loops using DFS
    all_loops = []
    visited_global = set()
    
    def dfs_find_loops(start_node, current_path, visited_path):
        """Depth-first search to find cycles"""
        if len(current_path) > max_loop_length:
            return
            
        current_node = current_path[-1] if current_path else start_node
        
        if current_node not in graph:
            return
            
        for neighbor in graph[current_node]:
            if neighbor == start_node and len(current_path) >= 2:
                # Found a loop back to start
                loop = current_path + [neighbor]
                if loop not in all_loops and len(loop) >= 3:  # Minimum loop length
                    all_loops.append(loop[:])  # Copy the loop
                    print(f"🔄 Found loop: {' → '.join(loop)}")
                    
            elif neighbor not in visited_path and len(current_path) < max_loop_length:
                # Continue exploring
                new_path = current_path + [neighbor]
                new_visited = visited_path | {neighbor}
                dfs_find_loops(start_node, new_path, new_visited)
    
    # Search for loops starting from each node
    print(f"🔍 Starting loop search from each node...")
    for i, node in enumerate(graph.keys()):
        print(f"  Searching from node {i+1}/{len(graph)}: {node}")
        dfs_find_loops(node, [node], {node})
    
    print(f"✅ Loop search completed. Found {len(all_loops)} raw loops")
    
    # Remove duplicate loops (same loop in different starting positions)
    unique_loops = []
    for loop in all_loops:
        # Remove the closing node (last element that repeats the first)
        cycle = loop[:-1]
        
        # Normalize to the lexicographically smallest rotation. Reversing is NOT a
        # valid normalization for a directed cycle: A->B->C->A and A->C->B->A are
        # different loops, and when the reversed order sorts smaller it is stored
        # as the loop, so every edges[(source, target)] lookup below misses, the
        # polarity product sees no "-" and the loop misreports as reinforcing.
        min_rotation = min(cycle[i:] + cycle[:i] for i in range(len(cycle)))

        if min_rotation not in unique_loops:
            unique_loops.append(min_rotation)
    
    print(f"🔄 After deduplication: {len(unique_loops)} unique loops")
    
    # Analyze loop polarities
    analyzed_loops = []
    for cycle in unique_loops:
        loop_edges = []
        loop_polarity = "+"
        
        # Create edges for the cycle (including closing edge)
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]  # Wrap around to close the loop
            edge_polarity = edges.get((source, target), "?")
            loop_edges.append({
                'source': source,
                'target': target, 
                'polarity': edge_polarity
            })
            
            # Calculate overall loop polarity (product of polarities)
            if edge_polarity == "-":
                loop_polarity = "+" if loop_polarity == "-" else "-"
        
        # Build ordered links list: [{from, to, polarity}] for dominance analysis
        ordered_links = []
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            ordered_links.append({
                'from': source,
                'to': target,
                'polarity': edges.get((source, target), '?')
            })

        analyzed_loops.append({
            'variables': cycle,  # The cycle variables
            'edges': loop_edges,
            'links': ordered_links,  # Ordered link sequence for loop score computation
            'length': len(cycle),
            'polarity': loop_polarity,
            'type': 'reinforcing' if loop_polarity == "+" else 'balancing'
        })
    
    # Sort loops by length, type, then the (rotation-normalized) variable tuple
    # so ordering — and the loop_N ids consumers assign from it — is fully
    # deterministic regardless of graph-traversal/discovery order, which can
    # vary with set iteration (PYTHONHASHSEED).
    analyzed_loops.sort(key=lambda x: (x['length'], x['type'], tuple(x['variables'])))
    
    return {
        'loops': analyzed_loops,
        'total_loops': len(analyzed_loops),
        'reinforcing_loops': len([l for l in analyzed_loops if l['type'] == 'reinforcing']),
        'balancing_loops': len([l for l in analyzed_loops if l['type'] == 'balancing']),
        'graph_nodes': list(graph.keys()),
        'graph_edges': causal_links,
        'adjacency_list': graph
    }


def create_cld_data_structure(causal_links, loop_analysis):
    """
    Create a comprehensive data structure for Causal Loop Diagram visualization
    """
    # Extract all unique variables
    all_variables = set()
    for link in causal_links:
        all_variables.add(link['source'])
        all_variables.add(link['target'])
    
    # Calculate centrality metrics
    in_degree = {}
    out_degree = {}
    for var in all_variables:
        in_degree[var] = len([l for l in causal_links if l['target'] == var])
        out_degree[var] = len([l for l in causal_links if l['source'] == var])
    
    # Identify key variables
    key_variables = []
    for var in all_variables:
        total_connections = in_degree[var] + out_degree[var]
        is_in_loop = any(var in loop['variables'] for loop in loop_analysis['loops'])
        
        key_variables.append({
            'name': var,
            'in_degree': in_degree[var],
            'out_degree': out_degree[var],
            'total_connections': total_connections,
            'in_loops': is_in_loop,
            'loop_count': sum(1 for loop in loop_analysis['loops'] if var in loop['variables'])
        })
    
    # Sort by importance (total connections + loop participation)
    key_variables.sort(key=lambda x: (x['total_connections'], x['loop_count']), reverse=True)
    
    return {
        'variables': key_variables,
        'relationships': causal_links,
        'loops': loop_analysis['loops'],
        'statistics': {
            'total_variables': len(all_variables),
            'total_relationships': len(causal_links),
            'total_loops': loop_analysis['total_loops'],
            'reinforcing_loops': loop_analysis['reinforcing_loops'],
            'balancing_loops': loop_analysis['balancing_loops'],
            'max_connections': max(key_variables, key=lambda x: x['total_connections'])['total_connections'] if key_variables else 0,
            'variables_in_loops': len([v for v in key_variables if v['in_loops']])
        },
        'ready_for_visualization': True
    }