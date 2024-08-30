## dictionary테스트 하기 getOrDefault

# dictionary = {"b" : 1}
#
# test_a = dictionary.get("a", 0)
# test_b = dictionary.get("b", 0)
#
# test_c = dictionary.get("a")
#
# print(test_a)
# print(test_b)
# print(test_c)

# want = ["banana", "apple", "apple", "pork", "pot"]
# want_dict = { }
#
# # for p in want:
# #     if want_dict.get(p):
# #         want_dict[p] += 1
# #     else:
# #         want_dict[p] = 1
#
# for p in want:
#     want_dict[p] = want_dict.get(p, 0) + 1
#
# for k, v in want_dict.items():
#     print(k, v)

## bfs 구현하기

# from collections import defaultdict, deque
# def solution(graph, start):
#     # bfs는 일단 연결된 노드를 바로 탐색한다.
#
#     # defaultdict에 노드 정보를 정렬해서 각 노드별로 연결된 노드를 정리하고
#     adj_list = defaultdict(list)
#     for u, v in graph:
#         adj_list[u].append(v)
#
#     # start를 queue에 넣고 visited와 result에도 start를 넣는다.
#     def bfs(start):
#         visited = set()
#         queue = deque([start])
#         visited.add(start)
#         result.append(start)
#         while queue:
#             node = queue.popleft()
#
#             for neighbor in adj_list[node]:
#                 if neighbor not in visited:
#                     queue.append(neighbor)
#                     visited.add(neighbor)
#                     result.append(neighbor)
#
#     # 현재 노드에 연결된 노드들을 순회하면서 visited가 아니면 visited와 result에 넣는다
#
#     # return할 result를 생성하고 bfs를 호출한다.
#
#     result = []
#     bfs(start)
#
#     return result

# print(solution([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9)],1)) # 반환값 :[1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(solution([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],1)) # 반환값 : [1, 2, 3, 4, 5, 0]

## 이진탐색트리 구현
#
# lst = [5, 3, 8, 4, 2, 1, 7, 10]
# search_lst = [1, 2, 5, 6]
#
#
# # lst = [1, 3, 5, 7, 9]
# # search_lst = [2, 4, 6, 8, 10]
#
# class Node:
#     def __init__(self, key):
#         self.left = None
#         self.right = None
#         self.val = key
#
#
# class BST:
#     def __init__(self):
#         self.root = None
#
#     def insert(self, key):
#         if not self.root:
#             self.root = Node(key)
#         else:
#             curr = self.root
#             while True:
#                 if key < curr.val:
#                     # curr.left가 없으면
#                     if curr.left:
#                         curr = curr.left
#                     else:  # curr.left가 있으면
#                         curr.left = Node(key)
#                         break
#                 else:
#                     if curr.right:
#                         curr = curr.right
#                     else:
#                         curr.right = Node(key)
#                         break
#
#     def search(self, key):
#         # search는 찾기만 해야함. root부터 비교해서 쭉 내려가면 됨
#         curr = self.root
#         while curr and curr.val != key:
#             if key < curr.val:
#                 curr = curr.left
#             else:
#                 curr = curr.right
#         return curr
#
# def solution(lst, search_lst):
#     answer = []
#
#     bst = BST()
#
#     # 이진트리에 집어넣기
#     for v in lst:
#         bst.insert(v)
#
#     # 이진트리 탐색하기
#     for search in search_lst:
#         if bst.search(search):
#             answer.append(True)
#         else:
#             answer.append(False)
#
#     return answer
#
# print(solution(lst, search_lst))

from collections import deque


def solution(maps):
    move = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    n = len(maps)
    m = len(maps[0])

    dist = [[-1] * m for _ in range(n)]

    def bfs(start):
        q = deque([start])
        dist[start[0]][start[1]] = 1  # 시작점을 1로

        while q:  # que가 유지되는동안 반복한다.
            here = q.popleft()  # 현위치 / 첫턴에는 [0, 0]

            # 제약조건 실행
            # 1. 4방위 이동시키기
            for direct in move:
                row, column = here[0] + direct[0], here[1] + direct[1]  # 현위치에 방위를 더하면 이동예정위치

                # 2. map을 벗어났는지 확인
                if row < 0 or column < 0 or row >= n or column >= m:
                    continue

                # 3. 벽인지 확인 0은 벽
                if maps[row][column] == 0:
                    continue

                # 4. 첫방문인지확인 => 첫방문이면 q에 추가하고 현위치보다 하나 높은값을 넣어놓기
                if dist[row][column] == -1:
                    q.append([row, column])
                    dist[row][column] = dist[here[0]][here[1]] + 1

    bfs([0, 0])
    return dist[n-1][m-1]


# map = [[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 1]]
map = [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,0],[0,0,0,0,1]]
print(solution(map))
