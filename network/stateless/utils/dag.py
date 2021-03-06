class DagNode:

    def __init__(self, name, edges = None, data = None):
        tmp_edges = edges if edges is not None else set()
        tmp_data = data if data is not None else {}
        assert isinstance(tmp_edges, set)
        assert isinstance(tmp_data, dict)
        self.name = name
        self.edges = tmp_edges
        self.data = tmp_data

    def add_edge(self, name):
        self.edges.add(name)

    def add_data(self, data):
        self.data.update(data)

    def get_edges(self):
        return self.edges.copy()

    def get_data(self):
        return self.data.copy()

    def assert_equal(self, other_node, data_equality_fn):
        assert self.name == other_node.name
        assert self.edges == other_node.edges
        assert data_equality_fn(self.data, other_node.data)
        

class Dag:

    def __init__(self):
        self.nodes = {}
        self.cache = {
            "ancestors": {},
            "descendants": {},
        }

    def remove_cache(self):
        self.cache = {
            "ancestors": {},
            "descendants": {},
        }

    def has_node(self, name):
        return name in self.nodes

    def add_node(self, name, edges=None, data=None):
        assert not self.has_node(name)
        self.remove_cache()
        self.nodes[name] = DagNode(name, edges, data)

    def has_no_circular_dependencies(self):
        edge_map = self.get_parent_map()
        all_nodes = self.get_node_names()
        for start_node in all_nodes:
            to_expand = [ start_node ]
            while len(to_expand) > 0:
                node = to_expand.pop()
                children = edge_map[node]
                for child in children:
                    if child == start_node:
                        return False
                    else:
                        to_expand.append(child)
        return True

    def add_edge(self, name, parent_name):
        assert name in self.nodes
        self.nodes[name].add_edge(parent_name)
        self.remove_cache()
        assert self.has_no_circular_dependencies()
    
    def add_data(self, name, data):
        assert name in self.nodes
        self.nodes[name].add_data(data)

    def get_node_names(self):
        return set(self.nodes.keys())

    def get_nodes(self):
        return set(self.nodes.items())

    def get_node(self, key):
        return self.nodes[key]

    def get_parent_map(self):

        if 'parent_map' in self.cache:
            return self.cache['parent_map']

        parent_map = {}
        for key, value in self.nodes.items():
            edges = value.get_edges()
            parent_map[key] = value.get_edges()
            for edge in edges:
                if edge not in parent_map:
                    parent_map[edge] = set()

        self.cache['parent_map'] = parent_map
        return parent_map

    def get_child_map(self):

        if 'child_map' in self.cache:
            return self.cache['child_map']

        child_map = {}
        for key, value in self.nodes.items():

            if key not in child_map:
                child_map[key] = set()

            for edge in value.get_edges():
                if edge not in child_map:
                    child_map[edge] = set([key])
                else:
                    if key not in child_map[edge]:
                        child_map[edge].add(key)

        self.cache['child_map'] = child_map
        return child_map

    def _get_dependencies(self, name, edge_map):
        assert name in self.nodes

        frontier = set([ name ]) 
        results = set()

        while len(frontier) > 0:
            node_name = frontier.pop()
            results.add(node_name)
            child_nodes = edge_map[node_name]
            for edge_name in child_nodes:
                if edge_name not in results and edge_name not in frontier:
                    frontier.add(edge_name)

        return results

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
        edge_map = self.get_parent_map()
        return edge_map[name]

    def get_children(self, name):
        edge_map = self.get_child_map()
        return edge_map[name]
    
    def get_ancestors(self, name):

        if name in self.cache['ancestors']:
            return self.cache['ancestors'][name]

        edge_map = self.get_parent_map()
        ancestors = self._get_dependencies(name, edge_map)

        self.cache['ancestors'][name] = ancestors
        return ancestors
    
    def get_ancestors_for_all(self, names):

        key = 'ancestors_for_all' + ''.join(sorted(names))
        if key in self.cache:
            return self.cache[key]

        edge_map = self.get_parent_map()
        ancestors_for_all = self._get_dependencies_arr(names, edge_map)

        self.cache[key] = ancestors_for_all
        return ancestors_for_all

    def get_descendants(self, name):

        if name in self.cache['descendants']:
            return self.cache['descendants'][name]

        edge_map = self.get_child_map()        
        descendants = self._get_dependencies(name, edge_map)

        self.cache['descendants'][name] = descendants
        return descendants
    
    def get_descendants_for_all(self, names):

        key = 'descendants_for_all' + ''.join(sorted(names))
        if key in self.cache:
            return self.cache[key]

        edge_map = self.get_child_map()
        descendants_for_all = self._get_dependencies_arr(names, edge_map)

        self.cache[key] = descendants_for_all
        return descendants_for_all

    def get_without_parents(self):
        return self._get_sourceless(self.get_parent_map())
    
    def get_without_descendants(self):
        return self._get_sourceless(self.get_child_map())

    def _topological_sort(self, edge_map):
        all_nodes = list(edge_map.keys())
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

        if 'ordered_from_top' in self.cache:
            return self.cache['ordered_from_top']

        ordered_from_top = self._topological_sort(self.get_parent_map())
        self.cache['ordered_from_top'] = ordered_from_top

        return ordered_from_top

    def ordered_from_bottom(self):

        if 'ordered_from_bottom' in self.cache:
            return self.cache['ordered_from_bottom']

        ordered_from_bottom = self._topological_sort(self.get_child_map())
        self.cache['ordered_from_bottom'] = ordered_from_bottom

        return ordered_from_bottom

    def merge_dag(self, other_dag, data_equality_fn):
        self.remove_cache()
        for key, value in other_dag.nodes.items():
            if key not in self.nodes:
                self.nodes[key] = value
            else:
                self.nodes[key].assert_equal(value, data_equality_fn)
    
