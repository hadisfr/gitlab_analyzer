#!/usr/bin/env python3

import argparse
from sys import argv

from graph_analyzer import GraphAnalyzer


def main():
    analyzer = GraphAnalyzer()

    args_parser = argparse.ArgumentParser(
        prog="%s" % argv[0],
        description="a GitLab.com network analyzer based on data from GitLab Crawler"
    )
    args_group = args_parser.add_mutually_exclusive_group(required=True)
    args_group.add_argument("--fork-chains", "-f", help="analyze chains of forks", action='store_true')
    args_group.add_argument("--bipartite", "-b", help="analyze bipartite graph of users-projects relations", action='store_true')
    args = vars(args_parser.parse_args())
    print("# GitLab Network", end="\n\n")
    if args['fork_chains']:
        analyzer.analyze_fork_chains()
    if args['bipartite']:
        analyzer.analyze_bipartite_graph()


if __name__ == '__main__':
    main()
