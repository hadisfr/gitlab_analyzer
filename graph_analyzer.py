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

    def plot_graph(self, graph, path, figsize, pos=None, node_color=None, label_key='label'):
        """Plot a graph to a file."""
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        if not node_color:
            color_pool = ['r', 'g', 'b', 'm']
            node_color = [color_pool[randrange(0, len(color_pool))] for node in graph.nodes]
        if not pos:
            pos = nx.drawing.nx_agraph.graphviz_layout(graph)
        nx.draw_networkx(graph, pos=pos, with_labels=True, labels=nx.get_node_attributes(graph, label_key),
                         node_size=9000, arrowsize=30, font_color='w', node_color=node_color)
        plt.savefig(os.path.join(os.path.dirname(__file__), "res", "%s.svg" % path))

    def save_graph(self, graph, path):
        """Save a grpah in graphml format."""
        nx.write_graphml(graph, os.path.join(os.path.dirname(__file__), "res", quote_plus("%s.graphml" % path)))

    def get_digraph_root(self, digraph):
        """Get root of a digraph."""
        return [node for node in digraph.nodes if digraph.in_degree(node) == 0][0]

    def analyze_fork_chains(self):
        """Analyze chains of forks."""
        def _analyze_centrality(graph, centrality, reverse=False):
            print("#### %s" % centrality, end="\n\n", flush=True)
            centralities = nx.__getattribute__(centrality)(graph.reverse() if reverse else graph)
            nx.set_node_attributes(graph, centralities, centrality)
            nodes_by_centrality = sorted(centralities.items(), key=lambda pair: pair[1], reverse=True)
            for i in range(len(nodes_by_centrality)):
                node = nodes_by_centrality[i]
                print("%d. %s (%f) with root %s" % (
                    i + 1,
                    graph.node[node[0]]['label'],
                    node[1],
                    graph.node[node[0]]['root']
                ), flush=True)
            print("", flush=True)

        def _analyze_longest_chain(graph, components):
            components_by_longest_path = [(nx.dag_longest_path_length(component), component) for component in components]
            components_by_longest_path.sort(key=lambda elm: elm[0], reverse=True)
            print("### Longest chain\n\nlength: %d" % components_by_longest_path[0][0] if len(components_by_longest_path) else 0,
                  end="\n\n", flush=True)
            for component in components_by_longest_path:
                if component[0] < components_by_longest_path[0][0]:
                    break
                root = self.get_digraph_root(component[1])
                print("* %s (%s)" % (graph.node[root]['label'], root), flush=True)
            print("", flush=True)

        print("## Fork Chains", end="\n\n", flush=True)
        graph = nx.DiGraph([(rel['source'], rel['destination']) for rel in self.db_ctrl.get_rows("forks")])
        nx.set_node_attributes(graph, self.get_projects_labels(graph), 'label')
        print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)), end="\n\n", flush=True)

        components = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
        components.sort(key=lambda component: len(component), reverse=True)
        print("### Component Size", end="\n\n", flush=True)
        for i in range(len(components)):
            component = components[i]
            root = self.get_digraph_root(component)
            print("%d. %s (%d)" % (
                i + 1,
                graph.node[root]['label'],
                len(component)
            ), flush=True)
            for node in component.nodes:
                graph.node[node]['root'] = root
        print("", flush=True)

        print("### Centrality", end="\n\n", flush=True)
        for centrality, reverse in [
            ('out_degree_centrality', False),
            ('eigenvector_centrality', True),
            ('katz_centrality', True)
        ]:
            _analyze_centrality(graph, centrality, reverse)

        self.save_graph(graph, self.output_files['fork_chains'])

        _analyze_longest_chain(graph, components)

        print("", flush=True)

    def get_projects_labels(self, nodes, sep='/'):
        """Get a dict of projects labels."""
        if isinstance(nodes, nx.classes.Graph):
            nodes = nodes.nodes
        return {row['id']: sep.join([row['owner_path'], row['path']]) for row in self.db_ctrl.get_rows_by_query(
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

    def analyze_bipartite_graph(self, rel_type="contributions"):
        """Analyze bipartite graph of users-projects relations."""
        def _user_to_id(user):
            return 'u%s' % user if user else user

        def _project_to_id(project):
            return 'p%s' % project if project else project

        def _id_to_user(node_id):
            return node_id[1:] if node_id else node_id

        def _id_to_project(node_id):
            return node_id[1:] if node_id else node_id

        def _create_graph():
            edges = [
                (_user_to_id(rel['user']), _project_to_id(rel['project']))
                for rel in self.db_ctrl.get_rows_by_query(
                    rel_type,
                    columns=['user', 'project']
                )
            ]
            graph = nx.Graph(edges)
            users = {edge[0] for edge in edges}
            projects = {edge[1] for edge in edges}
            nx.set_node_attributes(graph, {**{user: 'user' for user in users},
                                   **{project: 'project' for project in projects}}, 'type')
            nx.set_node_attributes(graph, {**{
                _user_to_id(user): label for (user, label) in self.get_users_labels([
                    _id_to_user(user) for user in users
                ]).items()
            }, **{
                _project_to_id(project): label for (project, label) in self.get_projects_labels([
                    _id_to_project(project) for project in projects
                ]).items()
            }}, 'label')
            self.save_graph(graph, self.output_files['bipartite'])
            print("n = %d (u = %d, p = %d), m = %d" % (len(graph.nodes), len(users), len(projects), len(graph.edges)),
                  end="\n\n", flush=True)
            return (graph, users, projects, edges)

        def _create_biclique_analyzer(edges):
            print("```", flush=True)
            biclique_analyzer = MaximalBicliques(
                input_addr=os.path.join(os.path.dirname(__file__), 'res',
                                        '%s_bipartite.txt' % self.output_files['bipartite']),
                output_addr=os.path.join(os.path.dirname(__file__), 'res',
                                         '%s.bicliques.txt' % self.output_files['bipartite']),
                output_size_addr=os.path.join(os.path.dirname(__file__), 'res',
                                              '%s.bicliques_size.txt' % self.output_files['bipartite']),
                store_temps=True
            )
            biclique_analyzer.calculate_bicliques([list(edge) for edge in edges])
            biclique_analyzer.bicliques.sort(key=lambda biclique: sorted((len(biclique[0]), len(biclique[1]))), reverse=True)
            print("```", flush=True)
            print("", flush=True)
            return biclique_analyzer

        print("## Bipartite Graph of Users-Projects Relations", end="\n\n", flush=True)
        (graph, users, projects, edges) = _create_graph()

        print("### Maximum Bicliques", end="\n\n", flush=True)
        biclique_analyzer = _create_biclique_analyzer(edges)

        for i in range(min(len(biclique_analyzer.bicliques), 10)):
            biclique_nodes = biclique_analyzer.bicliques[i]
            print("%d. %dx%d" % (i + 1, len(biclique_nodes[0]), len(biclique_nodes[1])))
            biclique = graph.subgraph(chain(*biclique_nodes))
            print("    * Users", flush=True)
            for user in users.intersection(biclique.nodes):
                print("        * %s" % graph.nodes[user]['label'])
            print("    * Projects", flush=True)
            for project in projects.intersection(biclique.nodes):
                print("        * %s" % graph.nodes[project]['label'])
            self.save_graph(biclique, "%s_%d" % (self.output_files['bipartite'], i + 1),)

        print("", flush=True)
