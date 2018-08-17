import json
import os.path
from sys import stderr
from urllib.parse import quote_plus
from random import randrange
from itertools import chain

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

    def plot_graph(self, graph, path, labels, figsize, pos=None, node_color=None):
        """Plot a graph to a file."""
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        if not node_color:
            color_pool = ['r', 'g', 'b', 'm']
            node_color = [color_pool[randrange(0, len(color_pool))] for node in graph.nodes]
        if not pos:
            pos = nx.drawing.nx_agraph.graphviz_layout(graph)
        nx.draw_networkx(graph, pos=pos, with_labels=True, labels=labels, node_size=9000, arrowsize=30,
                         font_color='w', node_color=node_color)
        plt.savefig(os.path.join(os.path.dirname(__file__), "res", "%s.svg" % path))

    def save_graph(self, graph, path):
        """Save a grpah in graphml format."""
        nx.write_graphml(graph, os.path.join(os.path.dirname(__file__), "res/%s" % path))

    def get_digraph_root(self, digraph):
        """Get root of a digraph."""
        return [node for node in digraph.nodes if digraph.in_degree(node) == 0][0]

    def analyze_fork_chains(self):
        """Analyze chains of forks."""
        print("## Fork Chains", end="\n\n", flush=True)
        graph = nx.DiGraph([(rel['source'], rel['destination']) for rel in self.db_ctrl.get_rows("forks")])
        print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)), end="\n\n", flush=True)

        components_by_longest_path = [(nx.dag_longest_path_length(component), component) for component in [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]]
        components_by_longest_path.sort(key=lambda elm: elm[0], reverse=True)

        print("### Centrality", end="\n\n", flush=True)
        for centrality in ['degree_centrality', 'eigenvector_centrality']:
            print("#### %s" % centrality, end="\n\n", flush=True)
            nodes_by_centrality = sorted(nx.__getattribute__(centrality)(graph).items(), key=lambda pair: pair[1], reverse=True)
            labels = self.get_projects_labels([node[0] for node in nodes_by_centrality])
            for i in range(len(nodes_by_centrality)):
                node = nodes_by_centrality[i]
                print("%d. %s (%f)" % (i + 1, labels[node[0]].replace('\n', '/'), node[1]), flush=True)
            for node in nodes_by_centrality[:10]:
                for component in (component[1] for component in components_by_longest_path):
                    if node[0] in component.nodes:
                        component_labels = self.get_projects_labels(component)
                        node_label = component_labels[node[0]].replace('\n', '/')
                        self.plot_graph(
                            component,
                            "%s_%s" % (self.output_files['fork_chains'], quote_plus(node_label)),
                            component_labels,
                            (100, 100),
                            node_color=['g' if n != node else 'k' for n in component.nodes]
                        )
            print("", flush=True)

        print("### Longest chain\n\nlength: %d" % components_by_longest_path[0][0] if len(components_by_longest_path) else 0,
              end="\n\n", flush=True)
        for component in components_by_longest_path:
            if component[0] < components_by_longest_path[0][0]:
                break
            labels = self.get_projects_labels(component[1])
            root = self.get_digraph_root(component[1])
            root_label = labels[root].replace('\n', '/')
            print("* %s" % root_label, flush=True)
            self.plot_graph(
                component[1],
                "%s_%s" % (self.output_files['fork_chains'], quote_plus(root_label)),
                labels,
                (20, 20),
                node_color=['b' if node != root else 'k' for node in component[1].nodes]
            )
        print("", flush=True)

        self.save_graph(graph, self.output_files['fork_chains'])
        print("", flush=True)

    def get_projects_labels(self, nodes):
        """Get a dict of projects labels."""
        if isinstance(nodes, nx.classes.Graph):
            nodes = nodes.nodes
        return {row['id']: "%s\n%s" % (row['owner_path'], row['path']) for row in self.db_ctrl.get_rows_by_query(
            "projects",
            ["id", "owner_path", "path"],
            "id in (%s)" % ", ".join("%s" for i in range(len(nodes))),
            nodes
        )}

    def get_users_labels(self, nodes):
        """Get a dict of users labels."""
        if isinstance(nodes, nx.classes.Graph):
            nodes = nodes.nodes
        return {row['id']: row['username'] for row in self.db_ctrl.get_rows_by_query(
            "users",
            ["id", "username"],
            "id in (%s)" % ", ".join("%s" for i in range(len(nodes))),
            nodes
        )}

    def analyze_bipartite_graph(self):
        """Analyze bipartite graph of users-projects relations."""
        def user_to_id(user):
            return 'u%s' % user if user else user

        def project_to_id(project):
            return 'p%s' % project if project else project

        def id_to_user(node_id):
            return node_id[1:] if node_id else node_id

        def id_to_project(node_id):
            return node_id[1:] if node_id else node_id

        print("## Bipartite Graph of Users-Projects Relations", end="\n\n", flush=True)
        graph = nx.Graph([
            (user_to_id(rel['user']), project_to_id(rel['project']))
            for rel in self.db_ctrl.get_rows('membership', columns=['user', 'project'])
        ])
        print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)), end="\n\n", flush=True)

        print("### Maximum Bicliques", end="\n\n", flush=True)
        print("```", flush=True)
        biclique_analyzer = MaximalBicliques(
            input_addr=os.path.join('res', '%s_bipartite.txt' % self.output_files['bipartite']),
            output_addr=os.path.join('res', '%s.bicliques.txt' % self.output_files['bipartite']),
            output_size_addr=os.path.join('res', '%s.bicliques_size.txt' % self.output_files['bipartite']),
            store_temps=True
        )
        biclique_analyzer.calculate_bicliques([list(edge) for edge in graph.edges])
        biclique_analyzer.bicliques.sort(key=lambda biclique: len(biclique[0]) * len(biclique[1]), reverse=True)
        print("```", flush=True)
        print("", flush=True)

        for i in range(min(len(biclique_analyzer.bicliques), 10)):
            biclique_nodes = biclique_analyzer.bicliques[i]
            print("%d. %dx%d" % (i + 1, len(biclique_nodes[0]), len(biclique_nodes[1])))
            biclique = graph.subgraph(chain(*biclique_nodes))
            self.plot_graph(
                biclique,
                "%s_%d" % (self.output_files['bipartite'], i),
                {
                    **{user_to_id(user): label for (user, label) in self.get_users_labels(
                        [id_to_user(node) for node in biclique_nodes[0]]
                    ).items()},
                    **{project_to_id(project): label for (project, label) in self.get_projects_labels(
                        [id_to_project(node) for node in biclique_nodes[1]]
                    ).items()}
                },
                (20, 20),
                pos=nx.drawing.layout.bipartite_layout(biclique, biclique_nodes[0]),
                node_color=['r' if n in biclique_nodes[0] else 'b' for n in biclique.nodes]
            )

        self.save_graph(graph, self.output_files['bipartite'])
        print("", flush=True)
