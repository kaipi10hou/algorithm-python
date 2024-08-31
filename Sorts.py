import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 9)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)

# print(heap)
# print(heapq.heappop(heap))
# print(heap)
# print(heapq.heappop(heap))

## 계수정렬
## abc순으로 각 빈도만큼 출력되어야함
def solution(s):
    counts = [0] * 26

    for c in s:
        counts[ord(c) - ord("a")] += 1

    sorted_str = ""
    for i in range(26):
        sorted_str += chr(i + ord("a")) * counts[i]

    return sorted_str

# print(solution("hello")) # ehllo
# print(solution("algorithm")) # aghilmort

## 오름차순 정렬된 두 배열 정렬되게 합치기
def solution(arr1, arr2):
    answer = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            answer.append(arr1[i])
            i += 1
        else:
            answer.append(arr2[j])
            j += 1

    while i < len(arr1):
        answer.append(arr1[i])
        i += 1
    while j < len(arr2):
        answer.append(arr2[j])
        j += 1

    return answer
# print(solution([1, 2, 3], [4, 5, 6]))
# print(solution([1, 3, 5], [2, 4, 6]))


## strings를 인자로 받아서 정렬하는데 정렬기준이 n번째 char값
def solution(strings, n):
    return sorted(strings, key=lambda x: (x[n], x))


# print(solution(["sun", "bed", "car"], 1))
# print(solution(["abce", "abcd", "cdx"], 2))

def solution(n):
    digits = list(str(n))
    digits.sort(reverse=True)
    return int("".join(digits))

# print(solution(118372))

def solution(array, commands):
    answer = []
    for i in range(len(commands)):
        answer.append(sorted(array[commands[i][0]-1 : commands[i][1]])[commands[i][2]-1])
    return answer

print(solution([1, 5, 2, 6, 3, 7, 4], [[2, 5, 3], [4, 4, 1], [1, 7, 3]]))