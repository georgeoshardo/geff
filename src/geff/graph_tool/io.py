from __future__ import annotations

import os
from typing import TYPE_CHECKING

import graph_tool as gt
import numpy as np
import zarr

import geff
import geff.utils
from geff.metadata_schema import GeffMetadata

if TYPE_CHECKING:
    from pathlib import Path


def _get_gt_type(arr: zarr.Array | np.ndarray) -> str:
    """Infers the graph-tool property map type from a zarr array or numpy array.

    See https://graph-tool.skewed.de/static/docs/stable/_modules/graph_tool.html#PropertyMap

    """
    if arr.dtype.kind in ("U", "S", "O"):
        return "string"

    # TODO very bad just don't respect the dtype for now, this needs to be handled better!
    dtype_map = {
        "b": "bool",
        "i": "int64_t",
        "u": "uint64_t",
        "f": "double",
    }
    base_type = dtype_map.get(arr.dtype.kind)
    if not base_type:
        raise TypeError(f"Unsupported dtype for graph-tool: {arr.dtype}")

    # Handle vector types for multi-dimensional properties - I think this works?
    return f"vector<{base_type}>" if arr.ndim > 1 else base_type


def geff_id_to_gt_vertex(g: gt.Graph, original_id: float) -> gt.Vertex:
    """
    Get the graph-tool vertex corresponding to a given original geff ID.
    """
    original_ids = g.vp["gt_id_"].a
    gt_index = np.where(original_ids == original_id)[0]
    if len(gt_index) == 0:
        raise ValueError(f"Original ID {original_id} not found in graph.")
    return g.vertex(gt_index[0])  # can do int() in the output to get the index


def read_gt(path: Path | str, validate: bool = True) -> gt.Graph:
    """
    Read a geff file into a graph-tool graph.

    graph-tool nodes are sort just "in" the graph, so we keep the original IDs
    in a vertex property map named "gt_id_".

    Args:
        path (Path | str): The path to the root of the geff zarr file.
        validate (bool, optional): If True, validate the geff file before
            loading. Defaults to True.

    Returns:
        A graph-tool graph containing the data from the geff file.

    """
    path = os.path.expanduser(str(path))

    if validate:
        geff.utils.validate(path)

    group = zarr.open_group(path, mode="r")
    metadata = GeffMetadata.read(group)

    g = gt.Graph(directed=metadata.directed)
    for key, val in metadata:
        prop = g.new_graph_property("object")
        prop[g] = val
        g.graph_properties[key] = prop

    node_ids = group["nodes/ids"][:]
    g.add_vertex(n=len(node_ids))

    id_type = _get_gt_type(node_ids)
    g.vertex_properties["gt_id_"] = g.new_vp(id_type, vals=node_ids)

    for name in group.get("nodes/props", []):
        prop_group = group[f"nodes/props/{name}"]
        values = prop_group["values"]
        prop_map = g.new_vertex_property(_get_gt_type(values))

        if "missing" in prop_group.array_keys():  # Handle sparse properties
            missing_mask = prop_group["missing"][:]
            values_arr = values[:]
            for i, v in enumerate(values_arr):
                if not missing_mask[i]:
                    prop_map[i] = v.tolist() if isinstance(v, np.ndarray) else v
        else:  # Handle dense properties with a single fast assignment
            prop_map.a = values[:]

        g.vertex_properties[name] = prop_map

    if "edges" in group.group_keys():
        edge_ids = group["edges/ids"][:]

        id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        def map_edge_ids():
            for u_id, v_id in edge_ids:
                yield id_to_idx[u_id]
                yield id_to_idx[v_id]

        flat_gt_edges = np.fromiter(map_edge_ids(), dtype=np.int64, count=len(edge_ids) * 2)
        gt_edges = flat_gt_edges.reshape(-1, 2)

        g.add_edge_list(gt_edges)

        for name in group.get("edges/props", []):
            prop_group = group[f"edges/props/{name}"]
            values = prop_group["values"]
            prop_map = g.new_edge_property(_get_gt_type(values))

            if "missing" in prop_group.array_keys():  # Handle sparse properties
                missing_mask = prop_group["missing"][:]
                values_arr = values[:]
                for i, e in enumerate(g.edges()):
                    if not missing_mask[i]:
                        v = values_arr[i]
                        prop_map[e] = v.tolist() if isinstance(v, np.ndarray) else v
            else:  # Handle dense properties with a single fast assignment
                prop_map.a = values[:]

            g.edge_properties[name] = prop_map

    return g, metadata
