import json
import os.path
from sys import stderr

import networkx as nx
from matplotlib import use as matplotlib_select_backedn
matplotlib_select_backedn('Agg')  # noqa
from matplotlib import pyplot as plt

from db_ctrl import DBCtrl


class GraphAnalyzer():
    """Analyze GitLab network graph"""
    config_file = os.path.join(os.path.dirname(__file__), "config.json")

    def __init__(self):
        try:
            with open(self.config_file) as f:
                config = json.load(f)
                self.output_files = config['output']
        except Exception as ex:
            print("Config file (%s) error: %s\n" % (self.config_file, ex), file=stderr, flush=True)
            exit(1)

        self.db_ctrl = DBCtrl()

    def plot_graph(self, graph, path, figsize):
        """Plot a graph to a file."""
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        nx.draw_networkx(graph, with_labels=True)
        plt.savefig("res/%s.svg" % path)

    def analyze_fork_chains(self):
        """Analyze chains of forks."""
        graph = self.get_forks_graph()
        print("## Fork Chains")
        print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)))

        max_longest_path = (0, [])
        for component in (graph.subgraph(c) for c in nx.weakly_connected_components(graph)):
            root = [node for node in component.nodes if component.in_degree(node) == 0][0]
            longest_path_length = nx.dag_longest_path_length(component)
            if longest_path_length > max_longest_path[0]:
                max_longest_path = (longest_path_length, [root])
            elif longest_path_length == max_longest_path[0]:
                max_longest_path[1].append(root)
        print("Longest chain: length: %d, root of components: %s" % (max_longest_path))

        self.plot_graph(graph, self.output_files['fork_chains'], (30, 30))
        print("")

    def get_forks_graph(self):
        """Get forks graph from DB."""
        graph = nx.DiGraph()
        for rel in self.db_ctrl.get_rows("forks", values={"source": 13083}):
            graph.add_node(rel['source'])
            graph.add_node(rel['destination'])
            graph.add_edge(rel['source'], rel['destination'])
        return graph
