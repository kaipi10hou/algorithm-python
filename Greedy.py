# 거스름돈. 동전의종류는 100, 50, 10, 1이다
# 화폐간의 관계가 배수라면 그리디 사용가능
def solution(amount):
    denominations = [100, 50, 10, 1]

    change = []
    for coin in denominations:
        while amount >= coin:
            change.append(coin)
            amount -= coin

    return change


# print(solution(123))
# print(solution(350))


# 부분배낭문제
def calculate_unit_value(items):
    for item in items:
        item.append(item[1] / item[0])
    return items

def sort_by_unit_value(items):
    items.sort(key=lambda x: x[2], reverse=True)
    return items

def knapsack(items, weight_limit):
    total_value = 0
    remaining_weight = weight_limit

    for item in items:
        if item[0] <= remaining_weight:
            total_value += item[1]
            remaining_weight -= item[0]
        else:
            fraction = remaining_weight / item[0]
            total_value += item[1] * fraction
            break
    return total_value

def solution(items, weight_limit):
    items = calculate_unit_value(items)
    items = sort_by_unit_value(items)

    return knapsack(items, weight_limit)


print(solution([[10, 19], [7, 10], [6, 10]], 15))
print(solution([[10, 60], [20, 100], [30, 120]], 50))
