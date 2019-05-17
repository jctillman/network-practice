
class DagNode:

    def __init__(self, name, edges, data):
        self.name = name
        self.edges = edges if edges is not None else [] 
        self.data = data if data is not None else {}

    def add_edge(self, name):
        self.edges.append(name)

    def add_data(self, data):
        self.data.update(data)

    def get_edges(self):
        return self.edges.copy()

    def get_data(self):
        return self.data

    def merge_node(self, other_node, equality_fn):
        assert self.name == other_node.name
        assert all([
            x == y for x, y in
            zip( self.edges, other_node.edges)
        ]) 
        assert equality_fn(self.data, other_node.data)
        

class Dag:

    def __init__(self):
        self.nodes = {}

    def has_node(self, name):
        return name in self.nodes

    def add_node(self, name, edges=None, data=None):
        assert not self.has_node(name)
        self.nodes[name] = DagNode(name, edges, data)

    def add_edge(self, name, upstream_name):
        assert name in self.nodes
        self.nodes[name].add_edge(upstream_name)
    
    def add_data(self, name, data):
        assert name in self.nodes
        self.nodes[name].add_data(data)

    def get_node_names(self):
        return self.nodes.keys()

    def get_nodes(self):
        return self.nodes.items()

    def get_node(self, key):
        return self.nodes[key]

    def get_edge_map(self):
        edge_map = {}
        for key, value in self.nodes.items():
            edge_map[key] = value.get_edges()
        return edge_map

    def get_edge_map_inverted(self):
        edge_map_inverted = {}
        for key, value in self.nodes.items():

            if key not in edge_map_inverted:
                edge_map_inverted[key] = []

            for edge in value.get_edges():
                if edge not in edge_map_inverted:
                    edge_map_inverted[edge] = [key]
                else:
                    if key not in edge_map_inverted[edge]:
                        edge_map_inverted[edge].append(key)

        return edge_map_inverted

    def _get_dependencies(self, name, edge_map):
        assert name in self.nodes

        frontier_names = [ name ] 
        results_names = []

        while len(frontier_names) > 0:
            node_name = frontier_names.pop()
            results_names.append(node_name)
            parent_nodes = edge_map[node_name]
            for edge_name in parent_nodes:
                if edge_name not in results_names:
                    frontier_names.append(edge_name)

        return results_names

    def _get_dependencies_arr(self, names, edge_map):
        return_value = set()
        for name in names:
            dep = self._get_dependencies(name, edge_map)
            return_value.update(set(dep))
        return list(return_value)
    
    def _get_sourceless(self, edge_map):
        returned = []
        for key, value in edge_map.items():
            if len(value) == 0:
                returned.append(key)
        return returned

    def get_parents(self, name):
        edge_map = self.get_edge_map()
        return edge_map[name]

    def get_children(self, name):
        edge_map = self.get_edge_map_inverted()
        return edge_map[name]
    
    def get_upstream(self, name):
        edge_map = self.get_edge_map()
        return self._get_dependencies(name, edge_map)
    
    def get_upstreams(self, names):
        edge_map = self.get_edge_map()
        return self._get_dependencies_arr(names, edge_map)

    def get_downstream(self, name):
        edge_map = self.get_edge_map_inverted()        
        return self._get_dependencies(name, edge_map)
    
    def get_downstreams(self, names):
        edge_map = self.get_edge_map_inverted()
        return self._get_dependencies_arr(names, edge_map)

    def get_nothing_upstream(self):
        return self._get_sourceless(self.get_edge_map())
    
    def get_nothing_downstream(self):
        return self._get_sourceless(self.get_edge_map_inverted())

    def _topological_sort(self, edge_map):
        all_nodes = list(edge_map.keys())
        total_num = len(all_nodes)
        return_list = self._get_sourceless(edge_map)
        
        remaining = [ x for x in all_nodes if x not in return_list ]

        while (len(remaining) > 0):

            next_batch = [
                x for x in remaining
                if all([
                    y in return_list for y
                    in edge_map[x]
                ])
            ]
            return_list.extend(next_batch)
            remaining = [ x for x in all_nodes if x not in return_list ]

        return return_list

    def ordered_from_top(self):
        return self._topological_sort(self.get_edge_map())

    def merge_dag(self, other_dag, data_equality_fn):
        for key, value in other_dag.nodes.items():
            if key not in self.nodes:
                self.nodes[key] = value
            else:
                self.nodes[key].merge_node(value, data_equality_fn)
    
