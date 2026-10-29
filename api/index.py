"""
Supply-Chain Graph API  (Vercel serverless edition)
───────────────────────────────────────────────────
NetworkX builds the DAG (Suppliers → Parts → Products → Orders).

Vercel runs each request in a stateless serverless function, so nothing can
be kept in memory between requests.  The browser therefore owns the state:
it sends the list of currently-removed node ids (`removed`) with every call,
and the server rebuilds the graph from the original JSON minus those nodes.

Endpoints
─────────
GET  /api/graph                    → graph as {nodes, edges}   (?removed=A,B)
GET  /api/downstream/<node_id>     → downstream trace          (?removed=A,B)
GET  /api/impact/<node_id>         → delete preview            (?removed=A,B)
DELETE /api/node/<node_id>         → cascade delete; body {"removed": [...]}
POST /api/undo                     → rebuild graph; body {"removed": [...]}
POST /api/reset                    → original graph
"""

import json, pathlib, time
from flask import Flask, jsonify, request, send_from_directory
import networkx as nx

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "photon_sprint1_graph_data.json"
if not DATA_FILE.exists():                       # fallback: data next to this file
    DATA_FILE = pathlib.Path(__file__).resolve().parent / "photon_sprint1_graph_data.json"

app = Flask(__name__)

with open(DATA_FILE, encoding="utf-8") as f:
    RAW = json.load(f)


# ── Graph helpers ────────────────────────────────────────────────────────

def _build_graph(raw: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for category in ("suppliers", "parts", "products", "orders"):
        for node in raw["nodes"][category]:
            G.add_node(node["id"], **node)
    for edge_type in ("supplies", "used_in", "fulfills"):
        for edge in raw["edges"][edge_type]:
            G.add_edge(
                edge["source"],
                edge["target"],
                relation=edge["relation"],
                **{k: v for k, v in edge.items() if k not in ("source", "target", "relation")},
            )
    return G


def _graph_without(removed) -> nx.DiGraph:
    """Original graph minus the nodes the client has already deleted."""
    G = _build_graph(RAW)
    G.remove_nodes_from([n for n in removed if n in G])
    return G


def _removed_from_request() -> list:
    """Read the `removed` list from JSON body or ?removed=A,B query string."""
    removed = []
    body = request.get_json(silent=True) or {}
    if isinstance(body.get("removed"), list):
        removed = body["removed"]
    qs = request.args.get("removed", "")
    if qs:
        removed = removed + [s for s in qs.split(",") if s]
    return [str(x) for x in removed]


def _graph_to_json(G: nx.DiGraph) -> dict:
    nodes = [{**attrs, "id": nid} for nid, attrs in G.nodes(data=True)]
    edges = [{"source": s, "target": t, **a} for s, t, a in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


def _downstream(G: nx.DiGraph, start: str) -> set:
    if start not in G:
        return set()
    return set(nx.descendants(G, start))


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():  # only used when running locally; Vercel serves /public statically
    return send_from_directory(BASE_DIR / "public", "index.html")


@app.route("/api/graph")
def api_graph():
    return jsonify(_graph_to_json(_graph_without(_removed_from_request())))


@app.route("/api/downstream/<node_id>")
def api_downstream(node_id: str):
    G = _graph_without(_removed_from_request())
    return jsonify({"root": node_id, "downstream": list(_downstream(G, node_id))})


@app.route("/api/impact/<node_id>")
def api_impact(node_id: str):
    G = _graph_without(_removed_from_request())
    if node_id not in G:
        return jsonify({"error": f"Node {node_id} not found"}), 404

    node_attrs = dict(G.nodes[node_id])
    outgoing = [{"source": u, "target": v, "relation": d.get("relation", "")}
                for u, v, d in G.out_edges(node_id, data=True)]
    incoming = [{"source": u, "target": v, "relation": d.get("relation", "")}
                for u, v, d in G.in_edges(node_id, data=True)]

    downstream_nodes = _downstream(G, node_id)
    affected = downstream_nodes | {node_id}
    info = [{"id": n, "kind": G.nodes[n].get("kind", ""), "label": G.nodes[n].get("label", n)}
            for n in sorted(affected)]
    edges_to_remove = [{"source": u, "target": v, "relation": a.get("relation", "")}
                       for u, v, a in G.edges(data=True) if u in affected or v in affected]

    return jsonify({
        "node": {"id": node_id, **node_attrs},
        "outgoing_edges": outgoing,
        "incoming_edges": incoming,
        "downstream_nodes": list(downstream_nodes),
        "affected_nodes_details": info,
        "edges_to_remove": edges_to_remove,
        "total_nodes_affected": len(affected),
        "total_edges_affected": len(edges_to_remove),
        "time_complexity": "O(V + E)",
        "time_complexity_explanation": f"Traversal & cascade removal visits {len(affected)} vertices and inspects {len(edges_to_remove)} edges in O(V + E) time.",
    })


@app.route("/api/node/<node_id>", methods=["DELETE"])
def api_delete_node(node_id: str):
    """Cascade-delete a node. The client stores the returned `deleted` list for undo."""
    G = _graph_without(_removed_from_request())
    if node_id not in G:
        return jsonify({"error": f"Node {node_id} not found"}), 404

    attrs = dict(G.nodes[node_id])
    t0 = time.perf_counter()

    to_remove = _downstream(G, node_id) | {node_id}
    details = [{"id": n, "kind": G.nodes[n].get("kind", ""), "label": G.nodes[n].get("label", n)}
               for n in sorted(to_remove)]
    edges_removed = [{"source": u, "target": v, "relation": a.get("relation", "")}
                     for u, v, a in G.edges(data=True) if u in to_remove or v in to_remove]

    G.remove_nodes_from(list(to_remove))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    complexity = f"NetworkX BFS traversal & removal of {len(to_remove)} nodes and {len(edges_removed)} edges"

    return jsonify({
        "deleted_node_id": node_id,
        "deleted_label": attrs.get("label", node_id),
        "deleted_kind": attrs.get("kind", ""),
        "deleted": sorted(to_remove),
        "deleted_details": details,
        "edges_removed": len(edges_removed),
        "edges_removed_list": edges_removed,
        "elapsed_ms": round(elapsed_ms, 3),
        "time_complexity": "O(V + E)",
        "time_complexity_details": complexity,
        "graph": _graph_to_json(G),
    })


@app.route("/api/undo", methods=["POST"])
def api_undo():
    """Rebuild the graph from the `removed` list (client already popped the last delete)."""
    t0 = time.perf_counter()
    G = _graph_without(_removed_from_request())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return jsonify({"graph": _graph_to_json(G), "undo_elapsed_ms": round(elapsed_ms, 3)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    G = _build_graph(RAW)
    return jsonify({"graph": _graph_to_json(G), "node_count": G.number_of_nodes()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
