################################## 이진트리
# search_lst의 값이 lst에서 검색이 되는지 boolean값을 배열에 담아 리턴

class Node:
    def __init__(self, key):
        self.val = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        curr_node = self.root
        while True:
            if curr_node:
                if key <= curr_node.val:
                    if curr_node.left:
                        curr_node = curr_node.left
                    else:
                        curr_node.left = Node(key)
                        break
                else:
                    if curr_node.right:
                        curr_node = curr_node.right
                    else:
                        curr_node.right = Node(key)
                        break
            else:
                self.root = Node(key)
                break

    def search(self, key):
        curr_node = self.root
        while curr_node:
            if key == curr_node.val:
                return True
            else:
                if key < curr_node.val:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right

def solution(lst, search_lst):
    bst = BST()
    for node in lst:
        bst.insert(node)

    answer = []
    for node in search_lst:
        if bst.search(node):
            answer.append(True)
        else:
            answer.append(False)
    return answer
# print(solution([5, 3, 8, 4, 2, 1, 7, 10], [1, 2, 5, 6])) # 반환값: [True, True, True, False]
# print(solution([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])) # 반환값: [False, False, False, False, False]

################################## union-find
# 노드의 개수 k와 명령어와 노드가 배열에 담긴 operations
def find(parents, x):
    if parents[x] == x:
        return x
    parents[x] = find(parents, parents[x])
    return parents[x]

def union(parents, x, y):
    root1 = find(parents, x)
    root2 = find(parents, y)

    parents[root2] = root1

def solution(k, operations):
    parents = [x for x in range(k)]
    for op in operations:
        if op[0] == 'u':
            union(parents, op[1], op[2])
        elif op[0] == 'f':
            find(parents, op[1])

    return len(set(find(parents, x) for x in range(k)))
# print(solution(3, [['u', 0, 1], ['u', 1, 2], ['f', 2]])) # 반환값: 1
# print(solution(4, [['u', 0, 1], ['u', 2, 3], ['f', 0]])) # 반환값: 2


################################# 깊이우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

from collections import defaultdict
def solution(graph, start):
    #깊이우선은 스택이다!
    dd = defaultdict(list)
    for u, v in graph:
        dd[u].append(v)

    def dfs(node):
        visited.add(node)
        answer.append(node)
        for neighbor in dd[node]:
            if neighbor not in visited:
                dfs(neighbor)

    visited = set()
    answer = []
    dfs(start)

    return answer

# print(solution([['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']], 'A')) # 반환값: ['A', 'B', 'C', 'D', 'E']
# print(solution([['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F']], 'A')) # 반환값: ['A', 'B', 'D', 'E', 'F', 'C']


################################## 너비우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

from collections import defaultdict, deque
def solution(graph, start):
    dd = defaultdict(list)
    for u, v in graph:
        dd[u].append(v)

    def bfs(start):
        q = deque([start])
        visited.add(start)
        answer.append(start)
        while q:
            node = q.popleft()
            for neighbor in dd[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    answer.append(neighbor)
                    q.append(neighbor)


    visited = set()
    answer = []
    bfs(start)
    return answer

# print(solution([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9)],1)) # 반환값 :[1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(solution([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],1)) # 반환값 : [1, 2, 3, 4, 5, 0]


################################## 백트래킹
# 1~n 까지 조합하여 10이 되는 조합의 리스트를 리턴해야한다.
def solution(n):
    answer = []
    def backtrack(sum, num_list, start):
        if sum == 10:
            answer.append(num_list)

        for i in range(start, n+1):
            if sum <= 10:
                backtrack(sum + i, num_list + [i], i + 1)

    backtrack(0, [], 1)
    return answer
print(solution(5)) # result = [[1, 2, 3, 4], [1, 4, 5], [2, 3, 5]]
print(solution(2)) # result = []
print(solution(7)) # result = [[1, 2, 3, 4], [1, 2, 7], [1, 3, 6], [1, 4, 5], [2, 3, 5], [3, 7], [4, 6]]
