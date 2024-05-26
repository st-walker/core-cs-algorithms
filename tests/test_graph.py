from collections import Counter
from unittest.mock import patch
from math import inf

import pytest

from calgo.graph import (UndirectedGraph,
                         DirectedGraph,
                         UnknownVertex,
                         contract_edge,
                         pick_random_edge,
                         find_min_cut,
                         bfs,
                         dfs,
                         connected_components,
                         shortest_path,
                         topo_sort,
                         reverse_graph,
                         find_sccs,
                         UndirectedWeightedGraph,
                         dijkstras)

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

@pytest.fixture
def search_graph_example():
    r"""
      1--3---5  7
     / \  \ /  /
    0---2--4  6   8

    i.e. 6 and 7 form own connected set
    and 8 is completely alone

    """

    g = UndirectedGraph(nvertices=9)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)
    g.add_edge(6, 7)

    return g

def test_bfs(search_graph_example):
    result = bfs(search_graph_example, 0)
    # Refer to diagram in fixture to see that this ordering is indeed
    # a breadth first search.
    assert result == [0, 1, 2, 3, 4, 5]

def test_dfs(search_graph_example):
    result = dfs(search_graph_example, 0)
    assert result == [0, 1, 3, 5, 4, 2]

def test_shortest_path(search_graph_example):
    distances = shortest_path(search_graph_example, 0)
    assert distances == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: inf, 7: inf, 8: inf}

def test_connected_components(search_graph_example):
    components = connected_components(search_graph_example)
    assert components == {frozenset([0, 1, 2, 3, 4, 5]),
                          frozenset([6, 7]),
                          frozenset([8])}

def test_topo_sort():
    r"""
      .->1----.
     /        |
    0         v
     \.->2--->3

    terrible diagram but 0 points to 1 and 2, each of which point to 3, which points to nothing
    """

    g = DirectedGraph(nvertices=4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)

    result = topo_sort(g)
    assert result == [0, 2, 1, 3]

@pytest.fixture
def scc_test_graph():
    """This is from the lectures"""
    g = DirectedGraph(nvertices=11)

    # First SCC (vertices 0, 1, 2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    # Second SCC (Just the vertex 3): Actually this is just connecting
    # the first SCC to the third SCC since this is a single-node SCC.
    g.add_edge(1, 3)
    g.add_edge(3, 4)
    g.add_edge(3, 5)

    # Third SCC (vertices 4, 5, 6)
    g.add_edge(4, 6)
    g.add_edge(6, 5)
    g.add_edge(5, 4)

    # Fourth SCC (vertices 7, 8, 9, 10)
    g.add_edge(9, 10)
    g.add_edge(10, 7)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    g.add_edge(10, 8)

    # Connecting 1st SCC to 4th SCC:
    g.add_edge(2, 10)
    g.add_edge(2, 9)

    # Connecting 4th SCC to third SCC:
    g.add_edge(10, 5)
    g.add_edge(7, 6)

    return g

@pytest.fixture
def scc_test_graph_three_triangles():
    g = DirectedGraph(nvertices=9)
    # Triangle loop 1
    g.add_edge(1, 0)
    g.add_edge(0, 2)
    g.add_edge(2, 1)

    # Connecting triangle 1 to triangle 2
    g.add_edge(3, 1)

    # Triangle loop 2
    g.add_edge(4, 3)
    g.add_edge(5, 4)
    g.add_edge(3, 5)

    # Connecting triangle 2 to triangle 3
    g.add_edge(6, 4)

    # Triangle loop 3
    g.add_edge(7, 6)
    g.add_edge(8, 7)
    g.add_edge(6, 8)

    return g


def test_find_sccs_three_connected_triangles(scc_test_graph_three_triangles):
    sccs = find_sccs(scc_test_graph_three_triangles)
    assert sccs == {frozenset([0, 1, 2]),
                    frozenset([3, 4, 5]),
                    frozenset([6, 7, 8])}


def test_find_sccs_other_example(scc_test_graph):
    sccs = find_sccs(scc_test_graph)
    assert sccs == {frozenset({3}),
                    frozenset({4, 5, 6}),
                    frozenset({7, 8, 9, 10}),
                    frozenset({0, 1, 2})}


def test_reverse_graph():
    g = DirectedGraph(nvertices=4)

    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    g.add_edge(1, 3)

    grev = reverse_graph(g)

    assert grev[0][2] == 1
    assert grev[1][0] == 1
    assert grev[2][1] == 1
    assert grev[3][1] == 1


def test_dijkstras():
    # Graph to be tested is a diamond like this:
    #
    #       1
    #      /|\
    #    1/ | \6
    #    /  |  \
    #   /   |   \
    #  0    |2   3
    #   \   |   /
    #   4\  |  /3
    #     \ | /
    #      \|/
    #       2
    #
    #       4


    g = UndirectedWeightedGraph(nvertices=4)
    g.add_edge(0, 1, w=1)
    g.add_edge(1, 2, w=2)
    g.add_edge(0, 2, w=4)
    g.add_edge(1, 3, w=6)
    g.add_edge(2, 3, w=3)
    g.add_vertex(4)


    distances = dijkstras(g, 0)
    # by inspection of the graph above we can see that
    # (0 -> 0) = 0
    # (0 -> 1) = 1
    # (0 -> 2) = 3
    # (0 -> 3) = 6
    # (0 -> 4) = inf
    # from IPython import embed; embed()
    assert distances == {0: 0,
                         1: 1,
                         2: 3,
                         3: 6,
                         4: inf}

def test_dijkstras_2():

    #   A-(5)->B-(5)-> C
    #   
    #  A->C  with weight 6

    # This is a good test example as if we
    # Greedily just always pick the next smallest weight,
    # We will end up with a path from A to C through B with distance 10.
    # Whereas if we just take the direct path with weight 6, we get the true
    # shortest path from A to C.

    g = UndirectedWeightedGraph(nvertices=3)
    g.add_edge(0, 1, w=5)
    g.add_edge(1, 2, w=5)
    g.add_edge(0, 2, w=6)

    d = dijkstras(g, 0)

    assert d == {0: 0,
                 1: 5,
                 2: 6}
