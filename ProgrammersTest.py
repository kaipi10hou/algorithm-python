# def solution(n, left, right):

#     answer = [0] * (n * n)
#     index = 0
#     for i in range(1,n+1):
#         for j in range(1, n+1):
#             if i <= j:
#                 answer[index] = j
#             elif i > j:
#                 answer[index] = i
#             index += 1
#
#     return answer[left: right + 1]
# def solution(n, left, right):
#     answer = []
#     for index in range(left, right + 1):
#         row = index // n
#         col = index % n
#         answer.append(max(row, col) + 1)
#     return answer
#
# print(solution(3, 2, 5))


# def solution(s):
#     answer = 0
#     n = len(s)
#     for i in range(n):
#         stack = []
#         for j in range(n):
#             c = s[(i + j) % n]
#             if c == '(' or c == '{' or c == '[':
#                 stack.append(c)
#             else:
#                 if not stack:
#                     break
#
#                 if c == ')' and stack[-1] == '(':
#                     stack.pop()
#                 elif c == '}' and stack[-1] == '{':
#                     stack.pop()
#                 elif c == ']' and stack[-1] == '[':
#                     stack.pop()
#         else:
#             if not stack:
#                 answer += 1
#     return answer
#
# print(solution("[](){}"))
#
# from collections import deque


#
# def solution(N, K):
#     queue = deque(range(1, N+1))
#
#     while len(queue) > 1:
#         for _ in range(K-1):
#             queue.append(queue.popleft())
#             queue.popleft()
#     return queue[0]
#
# print(solution(5,3))
#
# from math import ceil
# def solution(progresses, speeds):
#     answer = []
#
#     #완성까지남은 일수를 집합
#     # day_left = []
#     # for i, progress in enumerate(progresses):
#     #     day_left.append(ceil((100 - progress) / speeds[i]))
#     n = len(progresses)
#     day_left = [ceil((100 - progresses[i] / speeds[i])) for i in range(n)]
#
#
#     max_day = day_left[0]
#     count = 0
#
#     # 가장먼저배포되어야하는날짜와 비교
#     for i in range(0, len(day_left)):
#         if day_left[i] <= max_day:
#             count += 1
#         else:
#             answer.append(count)
#             count = 1
#             max_day = day_left[i]
#     answer.append(count)
#     return answer
#
# progresses = [95, 90, 99, 99, 80, 99]
# speeds = [1, 1, 1, 1, 1, 1]
# print(solution(progresses, speeds)) #[1, 3, 2]

# def count_sort(arr, target):
#     hashtable = [0] * (target + 1)
#
#     for num in arr:
#         if num <= target:
#             hashtable[num] = 1
#     return hashtable
#
#
# def solution(arr, target):
#     # arr의 특정 원소 두개를 더했을 때 target이 되느냐 마느냐
#     hashtable = count_sort(arr, target)
#     for num in arr:
#         complement = target - num
#         if (
#             complement != num
#             and complement > 0
#             and complement < target
#             and hashtable[complement] == 1
#         ):
#             return True
#     return False
#
#
# print(solution([1,2,3,4,8], 6))

# from collections import deque
#
# def solution(priorities, location):
#     que = deque()
#     for i, priority in enumerate(priorities):
#         que.append({i: priority})
#
#     # 반복 내용
#     priorities_set = set(priorities)
#     max_priority = max(priorities)
#     answer = 0
#
#     while True:
#         item_dict = que.popleft()
#         first_key = next(iter(item_dict))
#         value = item_dict[first_key]
#         if max_priority == value:
#             answer += 1
#             if len(priorities) > 0:
#                 priorities.remove(value)
#                 max_priority = max(priorities)
#             if first_key == location:
#                 break
#         else:
#             que.append(item_dict)
#
#     return answer
#
# print(solution([1, 1, 9, 1, 1, 1], 0))
#
# from collections import deque
#
#
# def solution(priorities, location):
#     answer = 0
#     que = deque([(i, p) for i, p in enumerate(priorities)])
#
#     while True:
#         cur = que.popleft()
#         if any(cur[1] < com[1] for com in que):
#             que.append(cur)
#         else:
#             answer += 1
#             if que[0] == location:
#                 return answer
# import collections
# def solution(participant, completion):
#     # # 참가자 dict을 만든다.
#     # part_dict = {}
#     # # dict으로 만드는 이유는 중복 숫자 1씩 올린다
#     # for part in participant:
#     #     if part in part_dict:
#     #         part_dict[part] += 1
#     #     else:
#     #         part_dict[part] = 1
#     # # 완주자를 반복돌면서 참가자 숫자 내린다
#     # for com in completion:
#     #     part_dict[com] -= 1
#     #
#     # # dict의 value가 0인 key를 돌아
#     # for key in part_dict.keys():
#     #     if part_dict[key] != 0:
#     #         return key
#     print(collections.Counter(participant))
#     print(collections.Counter(completion))
#     print(collections.Counter(participant) -collections.Counter(completion))
#
#
# print(solution(["leo", "kiki", "eden"], ["eden", "kiki"]))

# def solution(want, number, discount):
#     want_dict = {}
#     for i, w in enumerate(want):
#         want_dict[w] = number[i]
#
#     answer = 0
#
#     for i in range(len(discount) - 9):
#         discount_dict = {}
#         for j in range(i, i + 10):
#             if discount[j] in want:
#                 discount_dict[discount[j]] = discount_dict.get(discount[j], 0) + 1
#             else:
#                 break
#
#         if discount_dict == want_dict:
#             answer += 1
#     return answer
# print(solution(["banana", "apple", "rice", "pork", "pot"], [3, 2, 2, 2, 1],
#                ["chicken", "apple", "apple", "banana", "rice", "apple", "pork", "banana", "pork", "rice", "pot",
#                 "banana", "apple", "banana"]))

# def solution(record):
#     answer = []
#     uid = {}
#     for line in record:
#         cmd = line.split(" ")
#         if cmd[0] != "Leave":
#             uid[cmd[1]] = cmd[2]
#
#     for line in record:
#         cmd = line.split(" ")
#         if cmd[0] == "Enter":
#             answer.append(f"{uid[cmd[1]]}님이 들어왔습니다.")
#         elif cmd[0] == "Change":
#             pass
#         else:
#             answer.append(f"{uid[cmd[1]]}님이 나갔습니다.")
#
#     return answer
#
# print(solution(["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234","Enter uid1234 Prodo","Change uid4567 Ryan"]))

# # 트리순회
# nodes = [1, 2, 3, 4, 5, 6, 7]
# ### return = ["1 2 4 5 3 6 7", "4 2 5 1 6 3 7", "4 5 2 6 7 3 1"]
#
# def preorder(nodes, idx):
#     if idx < len(nodes):
#         ret = str(nodes[idx]) + ' '
#         ret += preorder(nodes, idx * 2 + 1)
#         ret += preorder(nodes, idx * 2 + 2)
#         return ret
#     else:
#         return ''
#
# def inorder(nodes, idx):
#     if idx < len(nodes):
#         ret = inorder(nodes, idx * 2 + 1)
#         ret += str(nodes[idx]) + ' '
#         ret += inorder(nodes, idx * 2 + 2)
#         return ret
#     else:
#         return ''
#
# def postorder(nodes, idx):
#     if idx < len(nodes):
#         ret = postorder(nodes, idx * 2 + 1)
#         ret += postorder(nodes, idx * 2 + 2)
#         ret += str(nodes[idx]) + ' '
#         return ret
#     else:
#         return ''
#
#
# def solution(nodes):
#     answer = []
#
#     answer.append(preorder(nodes, 0)[:-1])
#     answer.append(inorder(nodes, 0)[:-1])
#     answer.append(postorder(nodes, 0)[:-1])
#
#     print(answer)
#
#     return answer
#
# solution(nodes)

## 이진탐색트리 구현

# lst = [5, 3, 8, 4, 2, 1, 7, 10]
# search_lst = [1, 2, 5, 6]
# lst = [1, 3, 5, 7, 9]
# search_lst = [2, 4, 6, 8, 10]
#
# class Node:
#     def __init__(self, key):
#         self.val = key
#         self.left = None
#         self.right = None
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
#                     if curr.left:
#                         curr = curr.left
#                     else:
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
#         curr = self.root
#         while curr and curr.val != key:
#             if key < curr.val:
#                 curr = curr.left
#             else:
#                 curr = curr.right
#
#         return curr
#
# def solution(lst, search_lst):
#     bst = BST()
#     for key in lst:
#         bst.insert(key)
#
#     answer = []
#     for key in search_lst:
#         if bst.search(key):
#             answer.append(True)
#         else:
#             answer.append(False)
#
#     return answer
#
# print(solution(lst, search_lst))

# Union-Find

# operations = [['u', 0, 1], ['u', 1, 2], ['f,', 2]]
# k = 3

# operations = [['u', 0, 1], ['u', 2, 3], ['f', 0]]
# k = 4
#
# def find(parents, x):
#     # 집합에서 인덱스는 값 배열값은 부모노드
#
#     # 여기서 대입받은 x는 찾고자 하는 노드값
#     # find의 역할은 루트노드를 반납
#     # parents[x] 가 리턴하는 것은 부모노드의 값
#     # 그럼 find의 재귀가 끝나기 전까지 find는 부모노드의 반납을 계속함
#     # 부모노드의 끝에 루트노드가 있고 반납하고 끝.
#     if parents[x] == x: # 대입받은 값과 배열값이 같으면 루트노드이므로 반납
#         return x
#
#     parents[x] = find(parents, parents[x]) # 인덱스가 값이므로 값
#     return parents[x]
#
# def union(parents, x, y):
#     root1 = find(parents, parents[x])
#     root2 = find(parents, parents[y])
#
#     parents[root2] = root1
#
# def solution(k, operations):
#     answer = k
#     parents = list(range(k))
#
#     for op in operations:
#         if op[0] == 'u':
#             union(parents, op[1], op[2])
#         # if op[0] == 'f':
#         #     find(parents, op[1])
#
#     answer = len(set(find(parents, i) for i in range(k)))
#
#     return answer
#
#
# print(solution(k, operations))

# nums = [3,1,2,3]
# # nums = [3,3,3,2,2,4]
# # nums = [3,3,3,2,2,2]
#
# def solution(nums):
#     picks = len(nums) / 2
#     sorts = len(set(nums))
#
#     return min(sorts, picks)
#
#
# print(solution(nums))

# ##깊이우선탐색
# from collections import defaultdict
#
# def solution(graph, start):
#
#     # graph를 순회하면서 defaultdict에 넣어야해
#     adj_list = defaultdict(list)
#     for u, v in graph:
#         adj_list[u].append(v)
#
#     # dfs를 구현해야해
#     def dfs(node, visited, result):
#         visited.add(node)
#         result.append(node)
#         for neighbor in adj_list[node]:
#             if neighbor not in visited:
#                  dfs(neighbor, visited, result)
#
#     # dfs를 호출해야해
#     visited = set()
#     result = []
#     dfs(start, visited, result)
#
#     return result
#
#

# print(solution([['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']], 'A'))
# print(solution([['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F']], 'A'))

# ## 너비우선 탐색
# from collections import defaultdict, deque
#
#
# def solution(graph, start):
#     # 그래프를 인접 리스트로 변환
#     adj_list = defaultdict(list)
#     for u, v in graph:
#         adj_list[u].append(v)
#
#     # BFS 탐색 함수
#     def bfs(start):
#         visited = set()  # ❶ 방문한 노드를 저장할 셋
#
#         # ❷ 탐색시 맨 처음 방문할 노드 푸시 하고 방문처리
#         queue = deque([start])
#         visited.add(start)
#         result.append(start)
#
#         # ❸ 큐가 비어있지 않은 동안 반복
#         while queue:
#             node = queue.popleft()  # ❹ 큐에 있는 원소 중 가장 먼저 푸시된 원소 팝
#             for neighbor in adj_list.get(node, []):  # ❺  인접한 이웃 노드들에 대해서
#                 if neighbor not in visited:  # ❻ 방문되지 않은 이웃 노드인 경우
#                     # ❼ 이웃노드를 방문 처리함
#                     queue.append(neighbor)
#                     visited.add(neighbor)
#                     result.append(neighbor)
#
#     result = []
#     bfs(start)  # ❽ 시작 노드부터 BFS 탐색 수행
#     return result

# TEST 코드 입니다. 주석을 풀고 실행시켜보세요
# print(solution([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9)],1)) # 반환값 :[1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(solution([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],1)) # 반환값 : [1, 2, 3, 4, 5, 0]

from collections import deque


def solution(maps):
    move = [[-1, 0], [0, -1], [0, 1], [1, 0]]

    n = len(maps)
    m = len(maps[0])

    dist = [[-1] * m for _ in range(n)]

    def bfs(start):
        q = deque([start])
        dist[start[0]][start[1]] = 1

        while q:
            here = q.popleft()

            for direct in move:
                row, column = here[0] + direct[0], here[1] + direct[1]

                if row < 0 or row >= n or column < 0 or column >= m:
                    continue

                if maps[row][column] == 0:
                    continue

                if dist[row][column] == -1:
                    q.append([row, column])
                    dist[row][column] = dist[here[0]][here[1]] + 1

        # return dist

    bfs([0, 0])

    return dist[n - 1][m - 1]


# map = [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,1],[0,0,0,0,1]]
map = [[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 1]]


# print(solution(map))


### 네트워크(깊이우선탐색)

# solution은 컴퓨터(노드)의 개수 n과 연결상태를 나타내는 배열 computers를 파라미터로 한다.
def solution(n, computers):
    answer = 0

    # node에 방문을 했었는지 여부 배열
    visited = [False] * n

    # computers를 각 index 출발로 깊이우선탐색하며 첫방문 노드인 경우 answer + 1
    def dfs(node, computers, visited):
        visited[node] = True  # 방문처리
        for idx, connect in enumerate(computers[node]):
            if connect and not visited[idx]:  # [1, 1, 0] 에 대해서 1이면서, visited가 False면 깊이탐색
                dfs(idx, computers, visited)

    for i in range(n):
        if not visited[i]:
            dfs(i, computers, visited)
            answer += 1

    return answer


# print(solution(3, [[1, 1, 0], [1, 1, 0], [0, 0, 1]]	))

# https://school.programmers.co.kr/learn/courses/30/lessons/12949
def solution(arr1, arr2):
    r1, c1 = len(arr1), len(arr1[0])
    r2, c2 = len(arr2), len(arr2[0])

    answer = [[0] * c2 for _ in range(r1)]

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                answer[i][j] += arr1[i][k] * arr2[k][j]

    return answer


# print(solution([[1, 4], [3, 2], [4, 1]], [[3, 3], [3, 3]])) # [[15, 15], [15, 15], [15, 15]]
# print(solution([[2, 3, 2], [4, 2, 4], [3, 1, 4]]	, [[5, 4, 3], [2, 4, 1], [3, 1, 1]]	)) # [[22, 22, 11], [36, 28, 18], [29, 20, 14]]


# https://school.programmers.co.kr/learn/courses/30/lessons/132265?language=python3
def solution(topping):
    answer = 0
    for i in range(len(topping)):
        set1 = set(topping[:i])
        set2 = set(topping[i:])
        if len(set1) == len(set2):
            answer += 1
    return answer


# print(solution([1, 2, 1, 3, 1, 4, 1, 2])) # result = 2
# print(solution([1, 2, 3, 1, 4])) # result = 0
# ㄴ 시간복잡도에서 털린다

from collections import Counter


def solution(topping):
    answer = 0
    topping_dict = Counter(topping)
    topping_set = set()

    for t in topping:
        topping_dict[t] -= 1
        topping_set.add(t)
        if topping_dict[t] == 0:
            topping_dict.pop(t)
        if len(topping_dict) == len(topping_set):
            answer += 1

    return answer


# print(solution([1, 2, 1, 3, 1, 4, 1, 2])) # result = 2
# print(solution([1, 2, 3, 1, 4])) # result = 0


# https://school.programmers.co.kr/learn/courses/30/lessons/1844

from collections import deque

def solution(maps):
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n = len(maps)
    m = len(maps[0])

    dist = [[-1] * m for _ in range(n)]

    def bfs(start):
        q = deque([start])
        dist[start[0]][start[1]] = 1

        while q:
            here = q.popleft()

            for direct in move:
                row, column = here[0] + direct[0], here[1] + direct[1]

                if row < 0 or row >= n or column < 0 or column >= m:
                    continue

                if maps[row][column] == 0:
                    continue

                if dist[row][column] == -1:
                    q.append([row, column])
                    dist[row][column] = dist[here[0]][here[1]] + 1

        return dist

    bfs([0, 0])

    return dist[n-1][m-1]

print(solution([[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 1]]))  # 11
print(solution([[1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 1]]))  # -1
