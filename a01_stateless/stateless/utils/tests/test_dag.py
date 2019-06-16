import pytest

from stateless.utils.dag import Dag

def test_prevents_loop():
    a = Dag()
    for node in ['a','b','c']:
        a.add_node(node)
    a.add_edge('a', 'b')
    a.add_edge('b', 'c')
    with pytest.raises(Exception) as e:
        a.add_edge('c', 'a')

def test_edge_map_fncs():
    a = Dag()
    for node in ['a','b','c']:
        a.add_node(node)
    a.add_edge('a', 'b')
    a.add_edge('b', 'c')

    m = a.get_child_map()
    n = a.get_parent_map()
    assert len(m) == 3
    assert len(n) == 3

def test_get_dependencies():

    a = Dag()
    for node in ['a','b','c','d','e','f']:
        a.add_node(node)
    a.add_edge('c','a')
    a.add_edge('c','b')
    a.add_edge('d','c')
    a.add_edge('d','e')
    a.add_edge('f','d')

    assert set(a.get_ancestors('c')) == set(['a','b','c'])
    assert set(a.get_ancestors('d')) == set(['a','b','c', 'd', 'e'])
    
    assert set(a.get_descendants('c')) == set(['c','d','f'])
    assert set(a.get_descendants('a')) == set(['a', 'c','d','f'])
    assert set(a.get_descendants('f')) == set(['f'])
    assert set(a.get_descendants('d')) == set(['f', 'd'])


def test_basics():

    a = Dag()
    for node in ['a','b','c','d','e','f']:
        a.add_node(node)
    a.add_edge('c','a')
    a.add_edge('c','b')
    a.add_edge('d','c')
    a.add_edge('d','e')
    a.add_edge('f','d')



    assert set(a.get_without_parents()) == set(['a','b', 'e'])
    assert set(a.get_without_descendants()) == set(['f'])

    assert a.has_node('a') == True
    assert a.has_node('z') == False

