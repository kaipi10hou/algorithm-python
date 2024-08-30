## 깊이우선탐색
from collections import defaultdict
def solution(graph, start):
    adj_list = defaultdict(list)

    for u, v in graph:
        adj_list[u].append(v)

    def dfs(visited, result, node):
        visited.add(node)
        result.append(node)

        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dfs(visited, result, neighbor)


    visited = set()
    result = []
    dfs(visited, result, start)

    return result

# print(solution([['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']], 'A')) # 반환값: ['A', 'B', 'C', 'D', 'E']
# print(solution([['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F']], 'A')) # 반환값: ['A', 'B', 'D', 'E', 'F', 'C']

## 너비우선탐색

from collections import deque

def solution(graph, start):

    adj_list = defaultdict(list)
    for u, v in graph:
        adj_list[u].append(v)

    def bfs(start):
        visited = set()

        visited.add(start)
        result.append(start)
        q = deque([start])

        while q:
            node = q.popleft()
            for neighbor in adj_list.get(node, []): # adj_list[node]로 가져오면 값이 없을때 adj_list[node] = [] 로 추가해버린다
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    q.append(neighbor)

    result = []
    bfs(start)
    return result

# print(solution([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9)],1)) # 반환값 :[1, 2, 3, 4, 5, 6, 7, 8, 9]
print(solution([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],1)) # 반환값 : [1, 2, 3, 4, 5, 0]