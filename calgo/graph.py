from __future__ import annotations

import heapq
import logging
import os
import random
import tempfile
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from math import inf
from typing import Iterable, Iterator, TypeVar

import numpy as np

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class UnknownVertex(RuntimeError):
    pass


class GraphError(RuntimeError):
    pass


class DirectedGraph:
    def __init__(self, nvertices: int = 0):
        # Deliberately don't use defaultdict so __getitem__ doesn't create new vertices
        self._vertices = {}
        for ivertex in range(nvertices):
            # Initialize empty adjacency list for each vertex
            self._vertices[ivertex] = Counter()

    def add_edge(self, u: int, v: int) -> None:
        # Ask for permission instead of forgivess in this case because
        # Counter always returns for keys which don't exist (in in
        # which case it is 0).  But we are using our Counter indices
        # as to keep track of vertices as well.  But we want to handle
        # the case where one of the vertices doesn't exist, for added
        # robustness.  The user must explicitly add a vertex either at
        # init or with add_vertex.
        if u not in self._vertices:
            raise UnknownVertex(f"Vertex {u} does not exist in the graph")
        if v not in self._vertices:
            raise UnknownVertex(f"Vertex {v} does not exist in the graph")
        self._vertices[u][v] += 1

    def add_vertex(self, v: int) -> None:
        if v in self._vertices:
            raise ValueError(f"Vertex {v} already exists")
        self._vertices[v] = Counter()

    def pop_vertex(self, v: int) -> Counter:
        """Remove a vertex v from the Graph, and delete all edges
        featuring v, then Counter of outgoing (v, w) edges before the vertex was removed."""
        counter = self._vertices.pop(v)
        for vertex, count in counter.items():
            del self._vertices[vertex][v]
        return counter

    @property
    def nvertices(self) -> int:
        return len(self._vertices)

    def vertices(self) -> list[int]:
        return list(self._vertices.keys())

    @property
    def nedges(self) -> int:
        return sum ([counter.total() for counter in self._vertices.values()])

    def remove_self_loops(self) -> None:
        for u, adjacency_list in self._vertices.items():
            del adjacency_list[u]

    def count_self_loops(self) -> int:
        count = 0
        for u, adjacency_list in self._vertices.items():
            count += adjacency_list[u]
        return count

    def remove_edge(self, u: int, v: int) -> None:
        if u not in self._vertices:
            raise UnknownVertex(f"Vertex {u} does not exist in the graph")
        if v not in self._vertices:
            raise UnknownVertex(f"Vertex {v} does not exist in the graph")
        self._vertices[u][v] -= 1

    def __repr__(self):
        return f"<Graph: {self._vertices}>"

    def __getitem__(self, key: int | slice) -> list[int] | int:
        return self._vertices[key].copy() # Return a copy to prevent mutation

    def __contains__(self, key: int | tuple) -> bool:
        if isinstance(key, tuple):
            u, v = key
            try:
                return self._vertices[u][v] > 0
            except KeyError:
                return False
        return key in self._vertices

    def edges(self) -> Iterator[tuple[int, int]]:
        for u, adjacency_list in self._vertices.items():
            for v, count in adjacency_list.items():
                for _ in range(count):
                    yield u, v

    def adj(self) -> dict[int, tuple[int, int]]:
        pass

    def is_directed(self) -> bool:
        return True


class UndirectedGraph(DirectedGraph):
    """Class invariant: For each edge (u, v) there is an edge (v, u)"""
    def add_edge(self, u: int, v: int) -> None:
        super().add_edge(u, v)
        super().add_edge(v, u)

    def remove_edge(self, u: int, v: int) -> None:
        super().remove_edge(u, v)
        super().remove_edge(v, u)

    def __repr__(self):
        return f"<UndirectedGraph: {self._vertices}>"

    def edges(self) -> Iterator[tuple[int, int]]:
        # Returns edges, but opposite direction vertices are not
        # returned.  Basically we maintain class invariant that every
        # edge has a corresponding edge in the other direction, to
        # maintain the undirectedness of it.  If we have already seen
        # a given forward edge then we never yield later the
        # corresponding reverse edge.
        seen = set()
        for u, adjacency_list in self._vertices.items():
            for v, count in adjacency_list.items():
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                for _ in range(count):
                    yield u, v

    def is_directed(self) -> bool:
        return False

    @property
    def nedges(self) -> int:
        nedges_full = super().nedges
        assert (nedges_full % 2) == 0
        # We divide by two because we don't want to double count
        # forward and backward edges for our undirected graph.
        return int(nedges_full / 2)

    def pop_vertex(self, v: int) -> Counter:
        # self._assert_invariant()
        counter = super().pop_vertex(v)
        # Same as Undirected but because we are double counting the edges we divide them all by 2 here.
        # Class invariant is that in an undirected graph
        # self._assert_invariant({v: counter})
        return Counter({key: count for key, count in counter.items()})


GraphType = DirectedGraph | UndirectedGraph
GraphClass = type[GraphType]


class DirectedWeightedGraph:
    """Basic directed weighted graph for the sake of calculating Dijkstra's shortes path"""
    def __init__(self, nvertices: int = 0):
        # Deliberately don't use defaultdict so __getitem__ doesn't create new vertices
        self._vertices = {}
        for ivertex in range(nvertices):
            # Initialize empty adjacency list for each vertex
            self._vertices[ivertex] = []

    def add_edge(self, u: int, v: int, w: int = 1) -> None:
        # Ask for permission instead of forgivess in this case because
        # Counter always returns for keys which don't exist (in in
        # which case it is 0).  But we are using our Counter indices
        # as to keep track of vertices as well.  But we want to handle
        # the case where one of the vertices doesn't exist, for added
        # robustness.  The user must explicitly add a vertex either at
        # init or with add_vertex.
        if u not in self._vertices:
            raise UnknownVertex(f"Vertex {u} does not exist in the graph")
        if v not in self._vertices:
            raise UnknownVertex(f"Vertex {v} does not exist in the graph")
        self._vertices[u].append((v, w))

    def add_vertex(self, v: int) -> None:
        if v in self._vertices:
            raise ValueError(f"Vertex {v} already exists")
        self._vertices[v] = []

    @property
    def nvertices(self) -> int:
        return len(self._vertices)

    def vertices(self) -> list[int]:
        return list(self._vertices.keys())

    def remove_edge(self, u: int, v: int) -> None:
        if u not in self._vertices:
            raise UnknownVertex(f"Vertex {u} does not exist in the graph")
        if v not in self._vertices:
            raise UnknownVertex(f"Vertex {v} does not exist in the graph")
        self._vertices[u][v] -= 1

    def __repr__(self):
        return f"<Graph: {self._vertices}>"

    def __getitem__(self, key: int | slice) -> list[int] | int:
        return self._vertices[key].copy() # Return a copy to prevent mutation

    def __contains__(self, key: int | tuple) -> bool:
        if isinstance(key, tuple):
            u, v = key
            try:
                return self._vertices[u][v] > 0
            except KeyError:
                return False
        return key in self._vertices

    def edges(self) -> Iterator[tuple[int, int]]:
        for u, adjacency_list in self._vertices.items():
            for v, count in adjacency_list.items():
                for _ in range(count):
                    yield u, v

    def is_directed(self) -> bool:
        return True

    def is_weighted(self) -> bool:
        return True

    @classmethod
    def from_adj_list(cls, adj_list: dict[int, int]):
        g = cls()
        g._vertices = adj_list
        return g
    

class UndirectedWeightedGraph(DirectedWeightedGraph):
    def add_edge(self, u: int, v: int, w: int = 1) -> None:
        super().add_edge(u, v, w=w)
        super().add_edge(v, u, w=w)

    def is_directed(self) -> bool:
        return False


def load_undirected_graph_mincut_problem(fname: os.PathLike) -> UndirectedGraph:
    with open(fname, "r") as f:
        lines = f.readlines()

    graph = UndirectedGraph()

    rows = [list(map(int, line.split())) for line in lines]
    for row in rows:
        graph.add_vertex(row[0])

    seen = set()
    for row in rows:
        vertex_index, *adjacent_indices = row
        for adj in adjacent_indices:

            reverse = (adj, vertex_index)
            if reverse in seen:
                seen.remove(reverse)
                continue

            graph.add_edge(vertex_index, adj)
            seen.add((vertex_index, adj))

    return graph

def load_scc_problem_file(fname: str | os.PathLike) -> DirectedGraph:
    edges = np.loadtxt(fname, dtype=int)
    unique_vertices = np.unique(edges)
    assert np.unique(np.diff(unique_vertices)).item() == 1

    g = DirectedGraph(nvertices=len(unique_vertices))
    for u, v in edges:
        g.add_edge(u - 1, v - 1)

    return g


def load_graph_for_dijkstras(fname: str | os.PathLike) -> UndirectedWeightedGraph:
    path = os.fspath(fname)

    adj_list = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            u, *edges = line.split()
            u = int(u)
            for vw in edges:
                v, w = vw.split(",")
                v = int(v)
                w = int(w)
                adj_list[u].append((v, w))
    g = UndirectedWeightedGraph.from_adj_list(adj_list)
    return g

def pick_random_edge(graph: Graph) -> tuple[int, int]:
    edges = graph.edges()
    return random.choice(list(edges))

def contract_edge(graph: UndirectedGraph, u: int, v: int, self_loops=True):
    # Given an edge (u, v) contract, the edge by removing v and
    # merging all edges of v into u.

    # pop the vertex v from the graph, giving us all the vertices
    # adjacent to v, effectively now we have all the (v,w) edges.
    v_adjacents = graph.pop_vertex(v)

    # Edges (v, w), which will be attached to u instead, becoming edges (u, w).
    for w, wcount in v_adjacents.items():
        if not self_loops and u == w:
            continue
        for _ in range(wcount):
            graph.add_edge(u, w)

def find_min_cut(graph: UndirectedGraph, attempts: int = 100) -> int:
    """Returns the minimum number of cuts needed, not the identities
    of the edges themselves.  I could have done this but it would
    require more book keeping, probably a dedicated Edge class, to
    preserve the original identities of the edges after cutting
    begins.  Not interested in doing that.

    """
    min_cut = inf
    for _ in range(attempts):
        mc = _min_cut_attempt(graph)
        min_cut = min(min_cut, mc)
    return min_cut

def _min_cut_attempt(graph: UndirectedGraph) -> int:
    g = deepcopy(graph)
    if g.nvertices <= 2:
        raise GraphError("Graph has an insufficient number of vertices to find a meaningful minimum cut.")

    g.remove_self_loops()
    while g.nvertices > 2:
        u, v = pick_random_edge(g)
        contract_edge(g, u, v, self_loops=False)

    # The edges that are left over define a cut, hopefully minimum.
    return g.nedges


def bfs(graph: DirectedGraph | UndirectedGraph, u: int) -> list[int]:
    adj_initial = graph[u]
    # Read them in backwards so the ordering is nicer at the end...  Not necessary of course.
    queue = deque((v for v in reversed(adj_initial) if v!= u))

    # This is slower (a list) but I want testability in this codebase.  So order matters.
    seen = [u]
    while queue:
        v = queue.pop()
        if v in seen:
            continue
        v_adj = list(graph[v])
        queue.extendleft(v_adj)
        seen.append(v)

    return seen


def dfs(graph: DirectedGraph | UndirectedGraph, u: int) -> list[int]:
    adj_initial = graph[u]
    queue = deque((v for v in adj_initial if v!= u))

    # This is slower (a list) but I want testability in this codebase.  So order matters.
    seen = [u]
    while queue:
        v = queue.popleft()
        if v in seen:
            continue
        v_adj = list(graph[v])
        queue.extendleft(v_adj)
        seen.append(v)

    return seen

def shortest_path(graph: DirectedGraph | UndirectedGraph, u: int) -> dict[int, int]:
    adj_initial = graph[u]
    queue = deque(((u, v) for v in adj_initial if v!= u))
    # Set all distances to initially be zero, except for the starting point, which is by definition 0.
    distance = {v: inf for v in graph.vertices()}
    distance[u] = 0

    seen = {u}
    while queue:
        u, v = queue.pop()
        if v in seen:
            continue
        v_adj = list(graph[v])
        queue.extendleft((v, w) for w in v_adj)
        seen.add(v)
        distance[v] = distance[u] + 1

    return distance

def connected_components(graph: UndirectedGraph) -> set[frozenset[int]]:
    ccomponents = set()
    seen = set()
    for u in graph.vertices():
        u_components = frozenset(bfs(graph, u))
        seen |= u_components
        ccomponents.add(u_components)

    return ccomponents


def topo_sort(graph: DirectedGraph) -> list[int]:
    seen = set()
    rank = graph.nvertices
    # List of tuples, first entry is "rank", second is the vertex index
    ordering = []

    def _dfs_topo(graph: DirectedGraph, u: int):
        nonlocal rank # So we can do assigments to rank below.
        seen.add(u)

        for v in graph[u]:
            if v in seen:
                continue
            _dfs_topo(graph, v)

        ordering.append((rank, u))
        # ordering[u] = rank
        rank -= 1

    for u in graph.vertices():
        if u not in seen:
            _dfs_topo(graph, u)

    # Finally we sorted (recall sorting a list of tuples sorts by
    # first element) to get final ordering with respect to assigned
    # rank.  Yes this is nlogn and is therefore worse than the n+m
    # algorithm we are aiming for but I am not interested in
    # implementing a better search for now.
    ordering = sorted(ordering)
    # Pick second element (the vertex name):
    return [u for (_, u) in ordering]


def find_sccs(graph: DirectedGraph) -> set[frozenset[int]]:
    finishing_times = _scc_finishing_times(graph)

    # The vertex which we do the initial dfs whereby other nodes of
    # the SCC are discovered.
    current_leader = None
    seen = set()
    sccs = defaultdict(list)

    def _dfs(g: DirectedGraph, u: int):
        nonlocal current_leader
        sccs[current_leader]
        seen.add(u) # Mark u as now explored
        for v in g[u]:
            if v in seen:
                continue
            _dfs(g, v)
            sccs[current_leader].append(v)

    for timed_vertex in reversed(finishing_times):
        vertex = timed_vertex.vertex
        if vertex in seen:
            continue
        current_leader = vertex
        _dfs(graph, vertex)

    return set([frozenset([k, *v]) for k, v in sccs.items()])
    # return {frozenset({3, 4, 5}), frozenset({6, 7, 8}), frozenset({0, 1, 2})}

@dataclass
class FinishingTime:
    vertex: int
    time: int


def _scc_finishing_times(graph: DirectedGraph) -> list[FinishingTime]:
    grev = reverse_graph(graph)
    seen = set()
    finishing_times = [] # List of FinishingTime instances
    time = 0

    def _dfs(g: DirectedGraph, u: int):
        nonlocal time
        nonlocal finishing_times

        seen.add(u) # Mark u as now explored
        for v in g[u]:
            if v in seen:
                continue
            _dfs(g, v)

        finishing_times.append(FinishingTime(time=time, vertex=u))
        time += 1

    for u in grev.vertices():
        if u in seen:
            continue
        _dfs(grev, u)

    # Sort by time just in case...
    return finishing_times
    return sorted(finishing_times, key=lambda x: x.time)


def reverse_graph(graph: DirectedGraph) -> DirectedGraph:
    result = DirectedGraph(nvertices=graph.nvertices)
    for u, v in graph.edges():
        result.add_edge(v, u)
    return result


def dijkstras(graph: UndirectedWeightedGraph, start_node: int) -> dict[int, float | int]:
    distances = {start_node: 0}

    # List of dijkstra scores with edges, score first as we sort by
    # them frontier = the list of crossing edges with their weights if
    # we were to accept them.

    # Frontier is just a list of edges from visited nodes to unvisited
    # nodes with dijkstra scores.  
    frontier = [(w, start_node, v) for (v, w) in graph[start_node]]
    heapq.heapify(frontier)

    while frontier:
        next_edge = heapq.heappop(frontier)
        dscore, u, v = next_edge

        # If we have already computed the minimum distance for this
        # node then we just go to the next edge.  We don't mind adding
        # multiple edges to the same vertices, as the heap property
        # ensures we will always pick the one with the lowest dijkstra
        # score and subsequently we skip any edges to that node with
        # the following check:
        if v in distances:
            continue

        distances[v] = dscore
        # Get the already computed distance to the node u and add the
        # weight to get the minimum distance to w from start_node
        # Now we add all outgoing edges of v to the frontier heap with
        # their corresponding dijkstra scores.
        for t, t_weight in graph[v]:
            t_dscore = t_weight + dscore
            # If we haven't already calculated a minimum distance
            if t not in distances:
                heapq.heappush(frontier, (t_dscore, v, t))

    # Any undiscovered vertices simply get set to infinite distance.
    for v in graph.vertices():
        if v not in distances:
            distances[v] = inf

    return distances


def display_graph(graph: DirectedGraph | UndirectedGraph, prog="neato") -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    if graph.is_directed():
        G = nx.MultiDiGraph(graph.edges())
    elif not graph.is_directed():
        G = nx.MultiGraph(graph.edges())

    # Use PyGraphviz to draw the graph
    A = nx.nx_agraph.to_agraph(G)

    # Customize the graph's appearance
    A.node_attr['style'] = 'filled'
    A.node_attr['fillcolor'] = '#ffcccb'
    A.edge_attr['color'] = '#000000'
    # A.graph_attr['label'] = 'Graph with Parallel Edges and Self-loops'
    A.graph_attr['fontsize'] = '16'

    # Render and display the graph
    A.layout(prog=prog)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        A.draw(tmp.name, prog="circo")  # Save as PNG
        # Display the image in Python (optional)
        img = plt.imread(tmp.name)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
