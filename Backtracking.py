def solution(N):
    # 1~N 까지 조합하여 10이 되는 조합의 리스트를 리턴해야한다.
    result = []

    def backtrack(sum, num_list, start):
        if sum == 10:
            result.append(num_list)

        for i in range(start, N+1):
            if sum + i <= 10: # 전체합에 현재값을 더한 게 10보다 작으면 재귀
                backtrack(sum + i, num_list + [i], i+1) # 새로운전체합, 현재값이 추가된 조합, 이전값을 뺀 경우를 시작값으로 - (1부터 순차적이기때문에 가능)

    backtrack(0, [], 1)
    return result

# print(solution(5)) # result = [[1, 2, 3, 4], [1, 4, 5], [2, 3, 5]]
# print(solution(2)) # result = []
# print(solution(7)) # result = [[1, 2, 3, 4], [1, 2, 7], [1, 3, 6], [1, 4, 5], [2, 3, 5], [3, 7], [4, 6]]

## 던전피로도 문제
## https://school.programmers.co.kr/learn/courses/30/lessons/87946

def dfs(cur_k, cnt, dungeons, visited):
    answer_max = cnt
    for i in range(len(dungeons)):
        if cur_k >= dungeons[i][0] and visited[i] == 0:
            visited[i] = 1
            answer_max = max(
                answer_max, dfs(cur_k - dungeons[i][1], cnt + 1, dungeons, visited)
            )

            visited[i] = 0
    return answer_max


def solution(k, dungeons):
    visited = [0] * len(dungeons)
    answer_max = dfs(k, 0, dungeons, visited)
    return answer_max

# print( solution(80, [[80, 20], [50, 40], [30, 10]]) ) # answer = 3


def solution(k, dungeons):
    n = len(dungeons)
    answer_max = 0
    stack = [(k, 0, [0] * n)]  # (현재 체력, 던전 클리어 횟수, 방문 여부 리스트)

    while stack:
        cur_k, cnt, visited = stack.pop()
        answer_max = max(answer_max, cnt)

        for i in range(n):
            if cur_k >= dungeons[i][0] and visited[i] == 0:
                new_visited = visited[:]
                new_visited[i] = 1
                stack.append((cur_k - dungeons[i][1], cnt + 1, new_visited))

    return answer_max

# print( solution(80, [[80, 20], [50, 40], [30, 10]]) ) # answer = 3


# N퀸 https:///school.programmers.co.kr/learn/courses/30/lessons/12952
def getAns(n, y, width, diagonal1, diagonal2):
    ans = 0
    if y == n:
        ans += 1
    else:
        for i in range(n):
            if width[i] or diagonal1[i + y] or diagonal2[i - y + n]:
                continue
            width[i] = diagonal1[i + y] = diagonal2[i - y + n] = True
            ans += getAns(n, y + 1, width, diagonal1, diagonal2)
            width[i] = diagonal1[i + y] = diagonal2[i - y + n] = False
    return ans


def solution(n):
    ans = getAns(n, 0, [False] * n, [False] * (n * 2), [False] * (n * 2))
    return ans

print(solution(4)) # answer = 2
