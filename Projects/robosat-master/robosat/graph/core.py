import collections


class UndirectedGraph:
    '''Simple undirected graph.

    Note: stores edges; can not store vertices without edges.
    '''

    def __init__(self):
        '''Creates an empty `UndirectedGraph` instance.
        '''

        # Todo: We might need a compressed sparse row graph (i.e. adjacency array)
        # to make this scale. Let's circle back when we run into this limitation.
        self.edges = collections.defaultdict(set)

    def add_edge(self, s, t):
        '''Adds an edge to the graph.

        Args:
          s: the source vertex.
          t: the target vertex.

        Note: because this is an undirected graph for every edge `s, t` an edge `t, s` is added.
        '''

        self.edges[s].add(t)
        self.edges[t].add(s)

    def targets(self, v):
        '''Returns all outgoing targets for a vertex.

        Args:
          v: the vertex to return targets for.

        Returns:
          A list of all outgoing targets for the vertex.
        '''

        return self.edges[v]

    def vertices(self):
        '''Returns all vertices in the graph.

        Returns:
          A set of all vertices in the graph.
        '''

        return self.edges.keys()

    def empty(self):
        '''Returns true if the graph is empty, false otherwise.

        Returns:
          True if the graph has no edges or vertices, false otherwise.
        '''
        return len(self.edges) == 0

    def dfs(self, v):
        '''Applies a depth-first search to the graph.

        Args:
          v: the vertex to start the depth-first search at.

        Yields:
          The visited graph vertices in depth-first search order.

        Note: does not include the start vertex `v` (except if an edge targets it).
        '''

        stack = []
        stack.append(v)

        seen = set()

        while stack:
            s = stack.pop()

            if s not in seen:
                seen.add(s)

                for t in self.targets(s):
                    stack.append(t)

                yield s

    def components(self):
        '''Computes connected components for the graph.

        Yields:
          The connected component sub-graphs consisting of vertices; in no particular order.
        '''

        seen = set()

        for v in self.vertices():
            if v not in seen:
                component = set(self.dfs(v))
                component.add(v)

                seen.update(component)

                yield component
