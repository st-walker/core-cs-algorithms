from collections import Counter
from unittest.mock import patch

import pytest

from calgo.graph import UndirectedGraph, DirectedGraph, UnknownVertex, contract_edge, pick_random_edge, find_min_cut


@pytest.fixture
def undirected_tri_graph():
    g = UndirectedGraph(nvertices=3)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    return g


def test_DirectedGraph_init():
    g = DirectedGraph(nvertices=3)
    assert g.nvertices == 3
    assert not g[0]
    assert not g[1]
    assert not g[2]

def test_DirectedGraph_add_edge():
    g = DirectedGraph(nvertices=3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    assert list(g[0]) == [1]
    assert list(g[1]) == [2]
    assert list(g[2]) == [0]

def test_directedgraph_add_edge_raises_on_unknown_vertices():
    g = DirectedGraph(nvertices=1)
    with pytest.raises(UnknownVertex):
        g.add_edge(1,2)
    with pytest.raises(UnknownVertex):
        g.add_edge(2,1)
    with pytest.raises(UnknownVertex):
        g.add_edge(10, 20)

def test_directed_graph_add_vertex():
    g = DirectedGraph(nvertices=3)
    g.add_vertex(3)
    assert g.nvertices == 4
    assert g[3] == Counter()

def test_directed_graph_add_vertex_raises_if_duplicate_vertex():
    g = DirectedGraph(nvertices=1)
    with pytest.raises(ValueError):
        g.add_vertex(0)

def test_directed_graph_count_self_loops():
    g = DirectedGraph(nvertices=2)
    assert g.count_self_loops() == 0
    g.add_edge(0, 0)
    assert g.count_self_loops() == 1

def test_directed_graph_remove_self_loops():
    g = DirectedGraph(nvertices=2)
    assert g.count_self_loops() == 0
    g.add_edge(0, 0)
    assert g.count_self_loops() == 1
    g.remove_self_loops()
    assert g.count_self_loops() == 0    

def test_directed_graph_remove_edge():
    g = DirectedGraph(nvertices=2)
    g.add_edge(0, 1)
    assert (0, 1) in g
    g.remove_edge(0, 1)
    assert (0, 1) not in g

def test_undirected_graph_add_edge():
    g = UndirectedGraph(nvertices=2)
    g.add_edge(0, 1)
    assert (0, 1) in g
    assert (1, 0) in g

def test_undirected_graph_remove_edge():
    g = UndirectedGraph(nvertices=2)
    g.add_edge(0, 1)
    assert (0, 1) in g
    assert (1, 0) in g
    g.remove_edge(0, 1)
    assert (0, 1) not in g
    assert (1, 0) not in g


def test_pop_vertex():
    g = UndirectedGraph(nvertices=3)    
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    u0_adjacency_list = g.pop_vertex(0)
    assert (0, 1) not in g
    assert (1, 0) not in g
    assert (1, 2) in g
    assert (2, 1) in g
    assert u0_adjacency_list == Counter({1: 1, 2: 1})


def test_contract_edge(undirected_tri_graph):
    g = undirected_tri_graph
    contract_edge(g, 0, 1)

    assert g[0][2] == 2
    assert g[0][0] == 2 # A loop appears
    assert g[2][0] == 2

    
def test_contract_edge_with_remove_self_loops(undirected_tri_graph):
    g = undirected_tri_graph
    contract_edge(g, 0, 1, self_loops=False)

    assert g[0][2] == 2
    assert (0, 0) not in g
    assert g[2][0] == 2
    

@patch("random.choice")
def test_pick_random_edge(choice, undirected_tri_graph):
    g = undirected_tri_graph
    with patch("random.choice", choice):
        edge = pick_random_edge(g)
    assert choice.call_args.args == (list(g.edges()),)
    
def test_find_min_cut():
    """
    o---o
    |  /|
    | / |
    o---o
    """

    g = UndirectedGraph(nvertices=4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(1, 3)
    # Makes the above graph.

    min_cut = find_min_cut(g, attempts=10)
    assert min_cut == 2

