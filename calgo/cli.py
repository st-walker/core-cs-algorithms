import cProfile
import atexit
import pstats
import io

from click import option, group, argument, echo, Path, option

from calgo import quicksort
from calgo.graph import load_undirected_graph, find_min_cut


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
