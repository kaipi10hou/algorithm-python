## union-find 알고리즘
## 상호배타적 노드로 전제되어있음.
## k : 노드의 개수 / operation : 명령어와 노드
operations = [['u', 0, 1], ['u', 1, 2], ['f', 2]]
k = 3
# answer = 1

# operations = [['u', 0, 1], ['u', 2, 3], ['f', 0]]
# k = 4
## answer =2

def find(parents, x):
    # 해당 노드가 상위가 있는가?
    if parents[x] == x:
        return x
    parents[x] = find(parents, parents[x])
    return parents[x]


def union(parents, x, y):
    # union을 통해서 합쳐보자
    root1 = find(parents, x)
    root2 = find(parents, y)

    parents[root2] = root1


def solution(k, operations):
    # 각 노드를 배열로 초기화
    # k가 노드의 개수이므로
    parents = list(range(k))

    n = k

    for op in operations:
        if op[0] == 'u':
            union(parents, op[1], op[2])
        elif op[0] == 'f':
            find(parents, op[1])

    # 결과 총 부모의 개수는?
    return (len(set(find(parents, n) for n in range(k))))


print(solution(k, operations))
