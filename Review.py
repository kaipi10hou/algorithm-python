################################## 이진트리
# search_lst의 값이 lst에서 검색이 되는지 boolean값을 배열에 담아 리턴

# print(solution([5, 3, 8, 4, 2, 1, 7, 10], [1, 2, 5, 6])) # 반환값: [True, True, True, False]
# print(solution([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])) # 반환값: [False, False, False, False, False]

################################## union-find
# 노드의 개수 k와 명령어와 노드가 배열에 담긴 operations

# print(solution(3, [['u', 0, 1], ['u', 1, 2], ['f', 2]])) # 반환값: 1
# print(solution(4, [['u', 0, 1], ['u', 2, 3], ['f', 0]])) # 반환값: 2


################################# 깊이우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

# print(solution([['A', 'B'], ['B', 'C'], ['C', 'D'], ['D', 'E']], 'A')) # 반환값: ['A', 'B', 'C', 'D', 'E']
# print(solution([['A', 'B'], ['A', 'C'], ['B', 'D'], ['B', 'E'], ['C', 'F'], ['E', 'F']], 'A')) # 반환값: ['A', 'B', 'D', 'E', 'F', 'C']


################################## 너비우선탐색
# start를 시작노드로 graph탐색하는데 방문순서를 배열에 담아 리턴

# print(solution([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 9), (7, 9)],1)) # 반환값 :[1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(solution([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],1)) # 반환값 : [1, 2, 3, 4, 5, 0]


################################## 백트래킹
# 1~n 까지 조합하여 10이 되는 조합의 리스트를 리턴해야한다.

# print(solution(5)) # result = [[1, 2, 3, 4], [1, 4, 5], [2, 3, 5]]
# print(solution(2)) # result = []
# print(solution(7)) # result = [[1, 2, 3, 4], [1, 2, 7], [1, 3, 6], [1, 4, 5], [2, 3, 5], [3, 7], [4, 6]]

# N퀸

# print(solution(4)) # answer = 2


################################### 거스름돈
def solution(amount):
    denominations = [100, 50, 10, 1]
    answer = []
    return answer


# print(solution(123))  # answer = [100, 10, 10, 1, 1, 1]
# print(solution(350))  # answer = [100, 100, 100, 50]


################################## 부분배낭문제
# items는 [무게, 가치]가 배열로 담김, weight_limit은 배낭무게한도
def solution(items, weight_limit):
    answer = 0
    return answer


# print(solution([[10, 19], [7, 10], [6, 10]], 15))  # answer = 27.33
# print(solution([[10, 60], [20, 100], [30, 120]], 50))  # answer = 240


################################## 구명보트문제
# 사람들의 무게가 배열로 담긴 people, 보트의 제한무게 limit
def solution(people, limit):
    answer = 0
    return answer

# print(solution([70, 50, 80, 50], 100))  # answer = 3
# print(solution([70, 80, 50], 100))  # answer = 3