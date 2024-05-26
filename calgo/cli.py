import sys
import cProfile
import atexit
import pstats
import io

from click import option, group, argument, echo, Path, option

from calgo import quicksort
from calgo.graph import (load_undirected_graph_mincut_problem,
                         find_min_cut,
                         load_scc_problem_file,
                         find_sccs, dijkstras, load_graph_for_dijkstras)


@group()
@option("--profile", is_flag=True)
def main(profile):
    """Main entrypoint."""

    name = "calgo"
    echo(name)
    echo("=" * len(name))
    echo("Core Algorithms in Python")

    if profile:
        print("Profiling...")
        pr = cProfile.Profile()
        pr.enable()

        def exit():
            pr.disable()
            print("Profiling completed")
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats()
            print(s.getvalue())

        atexit.register(exit)




@main.command()
@argument("fname", nargs=1, type=Path(exists=True, dir_okay=False))
@option("--policy", default="random")
def qscount(fname, policy):
    """Quicksort with counting the number of comparisons"""
    with open(fname, "r") as f:
        lines = f.readlines()

    ints = [int(x) for x in lines]

    print(quicksort(ints, policy=policy))
    from IPython import embed; embed()


@main.command()
@argument("fname", nargs=1, type=Path(exists=True, dir_okay=False))
@option("--attempts", default=100)
def mincut(fname, attempts):
    graph = load_undirected_graph(fname)
    print(find_min_cut(graph, attempts=attempts))


@main.command()
@argument("fname", nargs=1)
def scc(fname):
    # Cheating a bit but life is short and I don't want to rewrite
    # algorithm to use deque+iteration instead of just recursion.
    sys.setrecursionlimit(500000)
    graph = load_scc_problem_file(fname)
    sccs = find_sccs(graph)
    lens = reversed(sorted([len(s) for s in sccs]))
    print(lens[:5])


@main.command()
@argument("fname", nargs=1)
def minpath(fname) -> None:
    g = load_graph_for_dijkstras(fname)
    distances = dijkstras(g, start_node=1)
    vertices_of_interest = [7,37,59,82,99,115,133,165,188,197]
    print(",".join([str(distances[v]) for v in vertices_of_interest]))
