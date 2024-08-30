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
print(solution([1, 2, 3], [4, 5, 6]))
print(solution([1, 3, 5], [2, 4, 6]))
