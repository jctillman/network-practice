
from stateless.utils.dag import Dag

def test_basics():

    a = Dag()
    for node in ['a','b','c','d','e','f']:
        a.add_node(node)
    a.add_edge('c','a')
    a.add_edge('c','b')
    a.add_edge('d','c')
    a.add_edge('d','e')
    a.add_edge('f','d')

    assert set(a.get_upstream('c')) == set(['a','b','c'])
    assert set(a.get_upstream('d')) == set(['a','b','c', 'd', 'e'])
    
    assert set(a.get_downstream('c')) == set(['c','d','f'])
    assert set(a.get_downstream('a')) == set(['a', 'c','d','f'])
    assert set(a.get_downstream('f')) == set(['f'])
    assert set(a.get_downstream('d')) == set(['f', 'd'])

    assert set(a.get_nothing_upstream()) == set(['a','b', 'e'])
    assert set(a.get_nothing_downstream()) == set(['f'])

    assert a.has_node('a') == True
    assert a.has_node('z') == False

