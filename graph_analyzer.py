import json
import os.path
from sys import stderr
from urllib.parse import quote_plus

import networkx as nx
from matplotlib import use as matplotlib_select_backedn
matplotlib_select_backedn('Agg')  # noqa
from matplotlib import pyplot as plt
from pybiclique import MaximalBicliques

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

    def plot_graph(self, graph, path, labels, figsize, pos=None):
        """Plot a graph to a file."""
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        if not pos:
            pos = nx.drawing.nx_agraph.graphviz_layout(graph)
        nx.draw_networkx(graph, pos=pos, with_labels=True, labels=labels, node_size=9000, arrowsize=30, font_color='w')
        plt.savefig(os.path.join(os.path.dirname(__file__), "res", "%s.svg" % path))

    def save_graph(self, graph, path):
        """Save a grpah in graphml format."""
        nx.write_graphml(graph, os.path.join(os.path.dirname(__file__), "res/%s" % path))

    def get_digraph_root(self, digraph):
        """Get root of a digraph."""
        return [node for node in digraph.nodes if digraph.in_degree(node) == 0][0]

    def analyze_fork_chains(self):
        """Analyze chains of forks."""
        graph = self.get_forks_graph()
        print("## Fork Chains", flush=True)
        print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)), flush=True)

        components_by_longest_path = [(nx.dag_longest_path_length(component), component) for component in [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]]
        components_by_longest_path.sort(key=lambda elm: elm[0], reverse=True)
        print("Longest chain (length: %d):" % components_by_longest_path[0][0] if len(components_by_longest_path) else 0,
              flush=True)

        for component in components_by_longest_path:
            if component[0] < components_by_longest_path[0][0]:
                break
            labels = self.get_projects_labels(component[1])
            root = labels[self.get_digraph_root(component[1])].replace('\n', '/')
            print("* %s" % root, flush=True)
            self.plot_graph(
                component[1],
                "%s_%s" % (self.output_files['fork_chains'], quote_plus(root)),
                labels, (20, 20)
            )
        print("", flush=True)

        self.save_graph(graph, self.output_files['fork_chains'])
        print("", flush=True)

    def get_forks_graph(self):
        """Get forks graph from DB."""
        graph = nx.DiGraph()
        for rel in self.db_ctrl.get_rows("forks"):
            graph.add_node(rel['source'])
            graph.add_node(rel['destination'])
            graph.add_edge(rel['source'], rel['destination'])
        return graph

    def get_projects_labels(self, graph):
        return {row['id']: "%s\n%s" % (row['owner_path'], row['path']) for row in self.db_ctrl.get_rows_by_query(
            "projects",
            ["id", "owner_path", "path"],
            "id in (%s)" % ", ".join("%s" for i in range(len(graph.nodes))),
            graph.nodes
        )}

    def analyze_bipartite_graph(self):
        """Analyze bipartite graph of users-projects relations."""
        print("## Bipartite Graph of Users-Projects Relations", flush=True)

        print("### Maximum Bicliques", flush=True)
        biclique_analyzer = MaximalBicliques(
            input_addr=os.path.join('res', 'bipartite.txt'),
            output_addr=os.path.join('res', 'bipartite.bicliques.txt'),
            output_size_addr=os.path.join('res', 'bipartite.bicliques_size.txt'),
            store_temps=True
        )
        biclique_analyzer.calculate_bicliques(
            [[rel['user'], rel['project']]
             for rel in self.db_ctrl.get_rows('membership', columns=['user', 'project'])]
        )
        biclique_analyzer.bicliques.sort(key=lambda biclique: len(biclique[0]) * len(biclique[1]), reverse=True)
