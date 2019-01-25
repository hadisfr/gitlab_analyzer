import json
import os.path
from sys import stderr
from urllib.parse import quote_plus
from random import randrange
from itertools import chain

import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
from matplotlib import use as matplotlib_select_backend
matplotlib_select_backend('Agg')  # noqa
from matplotlib import pyplot as plt
import seaborn as sns
from pybiclique import MaximalBicliques

from db_ctrl import DBCtrl

plt.rcParams['svg.fonttype'] = 'none'


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

    def distribution_analyze(self, data, attribute_name):
        """Analyze statistical distribution of data."""
        plt.clf()
        fit = powerlaw.Fit(data)
        print("Power Law with alpha = %f and standard error = %f" % (fit.power_law.alpha, fit.power_law.sigma))
        print("Lognormal with mu = %f and sigma = %f" % (fit.lognormal.mu, fit.lognormal.sigma))
        print("Power Law vs Lognormal: %s" % str(fit.distribution_compare('power_law', 'lognormal')))
        fit_fig = fit.plot_pdf(linewidth=3, label='Empirical Data')
        fit.lognormal.plot_pdf(ax=fit_fig, color='g', linestyle='--', label='Lognormal')
        fit.power_law.plot_pdf(ax=fit_fig, color='r', linestyle='--', label='PowerLaw')
        fit_fig.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), "res",
                                 quote_plus("%s_distro.svg" % attribute_name)))
        print("", flush=True)

    def get_fork_chains(self, is_verbose=False):
        """Get forest of forks."""
        if is_verbose:
            print("## Fork Chains", end="\n\n", flush=True)
        graph = nx.DiGraph([(rel['source'], rel['destination']) for rel in self.db_ctrl.get_rows("forks")])
        nx.set_node_attributes(graph, self.get_projects_labels(graph), 'label')
        if is_verbose:
            print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)), end="\n\n", flush=True)

        components = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
        components.sort(key=lambda component: len(component), reverse=True)
        trees_sizes = {}
        if is_verbose:
            print("### Components Size", end="\n\n", flush=True)
        for i in range(len(components)):
            component = components[i]
            root = self.get_digraph_root(component)
            trees_sizes[root] = (len(component))
            if is_verbose:
                print("%d. %s (%d)" % (
                    i + 1,
                    graph.node[root]['label'],
                    len(component)
                ), flush=True)
            for node in component.nodes:
                graph.node[node]['root'] = root
        return (graph, components, trees_sizes)

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

        (graph, components, trees_sizes) = self.get_fork_chains(is_verbose=True)
        print("", flush=True)

        print("#### Distribution", end="\n\n", flush=True)
        self.distribution_analyze(list(trees_sizes.values()), self.output_files['fork_chains'])

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
            biclique_analyzer.bicliques.sort(key=lambda biclique: len(biclique[0]) * len(biclique[1]), reverse=True)
            print("```", flush=True)
            print("", flush=True)
            return biclique_analyzer

        def _analyze_centrality(graph, centrality):
            print("##### %s" % centrality, end="\n\n", flush=True)
            centralities = nx.__getattribute__(centrality)(graph)
            nx.set_node_attributes(graph, centralities, centrality)
            nodes_by_centrality = sorted(centralities.items(), key=lambda pair: pair[1], reverse=True)
            for i in range(len(nodes_by_centrality)):
                node = nodes_by_centrality[i]
                print("%d. %s (%f)" % (
                    i + 1,
                    graph.node[node[0]]['label'],
                    node[1],
                ), flush=True)
            print("", flush=True)

        def _analyze_projected_graph(src, dst, name):
            print("### %s Graph" % name.capitalize(), end="\n\n", flush=True)
            # graph = nx.algorithms.bipartite.projected_graph(src, dst)
            graph = nx.algorithms.bipartite.weighted_projected_graph(src, dst)
            print("n = %d, m = %d" % (len(graph.nodes), len(graph.edges)),
                  end="\n\n", flush=True)
            self.save_graph(graph, "%s_%s" % (self.output_files['bipartite'], name),)

            print("#### Degree Distribution", end="\n\n", flush=True)
            self.distribution_analyze([d for n, d in graph.degree()],
                                      "%s_%s_degrees" % (self.output_files['bipartite'], name))

            print("#### Components Size Distribution", end="\n\n", flush=True)
            components = sorted([graph.subgraph(c) for c in nx.connected_components(graph)], key=len, reverse=True)
            print("%d components, max size = %d" % (len(components), len(components[0])))
            self.distribution_analyze([len(c) for c in components],
                                      "%s_%s_components_size" % (self.output_files['bipartite'], name))

            # diameter = max([nx.algorithms.distance_measures.diameter(component) for component in components])
            # diameter = nx.algorithms.distance_measures.diameter(components[0])  # TODO: remove weights, enhance time
            # print("diameter = %d" % diameter)

            # rich_club_coef = nx.rich_club_coefficient(graph)  # took a long time
            # print("Rich Club Coefficient = %d" % rich_club_coef)

            print("#### Centrality", end="\n\n", flush=True)
            for centrality in [
                'degree_centrality',
                'eigenvector_centrality',
                # 'katz_centrality'  # took a long time
            ]:
                _analyze_centrality(graph, centrality)

        print("## Bipartite Graph of Users-Projects Relations", end="\n\n", flush=True)
        (graph, users, projects, edges) = _create_graph()

        _analyze_projected_graph(graph, users, "users")
        _analyze_projected_graph(graph, projects, "projects")

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

    def draw_correlation_matrix(self, correlation_matrix, name):
        """Draw correlation matrix."""
        plt.figure(figsize=(8, 8))
        mask = np.zeros_like(correlation_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        heatmap = sns.heatmap(correlation_matrix, square=True, mask=mask, center=0.5, cmap="Spectral")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=-90)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        plt.savefig(os.path.join(os.path.dirname(__file__), "res", quote_plus("%s.svg" % name)))

    def analyze_projects_attributes(self):
        print("## Projects' Attributes", end="\n\n", flush=True)
        numerical_attributes = ["stars", "forks", "commit_count", "storage_size",
                                "repository_size", "lfs_objects_size"]
        binary_attributes = ["ci_config_path", "description", "avatar", "owned_by_user", "archived"]
        projects = {project['id']: project for project in
                    self.db_ctrl.get_rows("projects", columns=["id"] + numerical_attributes + binary_attributes)}
        numerical_attributes.append("forks_tree_size")
        for (project, forks) in self.get_fork_chains()[2].items():
            try:
                projects[project]["forks_tree_size"] = forks
            except KeyError:
                pass
        for project, row in projects.items():
            for attribute in binary_attributes:
                row[attribute] = True if row[attribute] and row[attribute] != "" and row[attribute] != 0 else False
        projects = pd.DataFrame(list(projects.values())).drop('id', axis='columns').fillna(0)

        print("### Pearson Correlation", end="\n\n", flush=True)
        pearson_correlation = projects.corr(method="pearson")
        print("```", flush=True)
        print(pearson_correlation.to_string())
        print("```", flush=True)
        self.draw_correlation_matrix(pearson_correlation, "%s_pearson_correlation"
                                     % self.output_files['projects_attributes'])

        print("### Spearman Correlation", end="\n\n", flush=True)
        spearman_correlation = projects.corr(method="spearman")
        print("```", flush=True)
        print(spearman_correlation.to_string())
        print("```", flush=True)
        self.draw_correlation_matrix(spearman_correlation, "%s_spearman_correlation"
                                     % self.output_files['projects_attributes'])

        print("### Distribution")
        for attribute in numerical_attributes:
            print("#### %s" % attribute)
            self.distribution_analyze(list(projects[attribute]),
                                      "%s_%s" % (self.output_files['projects_attributes'], attribute))
