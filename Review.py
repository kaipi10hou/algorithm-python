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
        if self.root:
            curr = self.root
            while True:
                if key <= curr.val:
                    if curr.left:
                        curr = curr.left
                    else:
                        curr.left = Node(key)
                        break
                else:
                    if curr.right:
                        curr = curr.right
                    else:
                        curr.right = Node(key)
        else:
            self.root = Node(key)

    def search(self, key):
        curr = self.root
        while curr and key != curr.val:
            if key <= curr.val:
                curr = curr.left
            else:
                curr = curr.right
        return curr


def solution(lst, search_lst):
    bst = BST()
    for n in lst:
        bst.insert(n)

    answer = []
    for key in search_lst:
        if bst.search(key):
            answer.append(True)
        else:
            answer.append(False)

    return answer


# print(solution([5, 3, 8, 4, 2, 1, 7, 10], [1, 2, 5, 6])) # 반환값: [True, True, True, False]
# print(solution([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])) # 반환값: [False, False, False, False, False]

################################## union-find
# 노드의 개수 k와 명령어와 노드가 배열에 담긴 operations

def find(arr, x):
    if arr[x] == x:
        return x
    arr[x] = find(arr, arr[x])
    return arr[x]


def union(arr, x, y):
    root1 = find(arr, x)
    root2 = find(arr, y)

    arr[root2] = root1


def solution(k, operations):
    arr = [i for i in range(k)]
    for op in operations:
        if op[0] == 'u':
            union(arr, op[1], op[2])
        elif op[0] == 'f':
            find(arr, op[1])
    return len(set(find(arr, x) for x in range(k)))


# print(solution(3, [['u', 0, 1], ['u', 1, 2], ['f', 2]])) # 반환값: 1
# print(solution(4, [['u', 0, 1], ['u', 2, 3], ['f', 0]])) # 반환값: 2


################################# 깊이우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

from collections import defaultdict


def solution(graph, start):
    adj_list = defaultdict(list)
    for u, v in graph:
        adj_list[u].append(v)

    answer = []
    visited = set()

    def dfs(node):
        visited.add(node)
        answer.append(node)
        for neighbor in adj_list.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return answer


# print(solution([['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']], 'A')) # 반환값: ['A', 'B', 'C', 'D', 'E']
# print(solution([['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F']], 'A')) # 반환값: ['A', 'B', 'D', 'E', 'F', 'C']


################################## 너비우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

from collections import deque


def solution(graph, start):
    adj_list = defaultdict(list)
    for u, v in graph:
        adj_list[u].append(v)

    answer = []
    visited = set()
    q = deque([start])

    answer.append(start)
    visited.add(start)

    while q:
        node = q.popleft()

        for neighbor in adj_list.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                answer.append(neighbor)
                q.append(neighbor)

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
        else:
            for i in range(start, n + 1):
                if sum <= 10:
                    backtrack(sum + i, num_list + [i], i + 1)

    backtrack(0, [], 1)
    return answer


# print(solution(5)) # result = [[1, 2, 3, 4], [1, 4, 5], [2, 3, 5]]
# print(solution(2)) # result = []
# print(solution(7)) # result = [[1, 2, 3, 4], [1, 2, 7], [1, 3, 6], [1, 4, 5], [2, 3, 5], [3, 7], [4, 6]]

# N퀸
def solution(n):
    def getAns(n, y, width, diagonal1, diagonal2):
        answer = 0
        if n == y:
            return 1
        for i in range(n):
            if width[i] or diagonal1[i + y] or diagonal2[i - y + n]:
                continue
            width[i] = diagonal1[i + y] = diagonal2[i - y + n] = True
            answer += getAns(n, y + 1, width, diagonal1, diagonal2)
            width[i] = diagonal1[i + y] = diagonal2[i - y + n] = False
        return answer

    return getAns(n, 0, [False] * n, [False] * n * 2, [False] * n * 2)


# print(solution(4)) # answer = 2


################################### 거스름돈
def solution(amount):
    denominations = [100, 50, 10, 1]
    answer = []
    for coin in denominations:
        while coin <= amount:
            answer.append(coin)
            amount -= coin
    return answer


# print(solution(123))  # answer = [100, 10, 10, 1, 1, 1]
# print(solution(350))  # answer = [100, 100, 100, 50]


################################## 부분배낭문제
# items는 [무게, 가치]가 배열로 담김, weight_limit은 배낭무게한도
def solution(items, weight_limit):
    answer = 0
    for item in items:
        item.append(item[1] / item[0])

    items.sort(key=lambda x: x[2], reverse=True)

    remain_weight = weight_limit
    for item in items:
        if remain_weight >= item[0]:
            remain_weight -= item[0]
            answer += item[1]
        else:
            answer += remain_weight / item[0] * item[1]
            break

    return answer


# print(solution([[10, 19], [7, 10], [6, 10]], 15))  # answer = 27.33
# print(solution([[10, 60], [20, 100], [30, 120]], 50))  # answer = 240


################################## 구명보트문제
# 사람들의 무게가 배열로 담긴 people, 보트의 제한무게 limit
def solution(people, limit):
    answer = 0
    a, b = 0, len(people) - 1
    people.sort()
    while a <= b:
        if people[a] + people[b] <= limit:
            a += 1
        b -= 1
        answer += 1

    return answer


# print(solution([70, 50, 80, 50], 100))  # answer = 3
# print(solution([70, 80, 50], 100))  # answer = 3


def solution(genres, plays):
    answer = []

    g_dic = {}
    p_dic = {}

    for i in range(len(genres)):
        genre = genres[i]
        play = plays[i]

        if genre not in g_dic:
            g_dic[genre] = []
            p_dic[genre] = 0

        g_dic[genre].append((i, play))
        p_dic[genre] += play

    sorted_pop = sorted(p_dic.items(), key=lambda x: x[1], reverse=True)

    for g, _ in sorted_pop:
        g_dic[g].sort(key=lambda x: x[1], reverse=True)
        answer.extend(i for i, _ in g_dic[g][:2])

    return answer


# print(solution(["classic", "pop", "classic", "classic", "pop"], [500, 600, 150, 800, 2500]))  # [4, 1, 3, 0]
print(solution(["pop", "pop", "classic", "classic", "pop"], [100, 100, 250, 250, 100]))  # [4, 1, 3, 0]
