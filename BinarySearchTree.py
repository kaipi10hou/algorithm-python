lst = [5, 3, 8, 4, 2, 1, 7, 10]
search_lst = [1, 2, 5, 6]
# lst = [1, 3, 5, 7, 9]
# search_lst = [2, 4, 6, 8, 10]

class Node:
    def __init__(self, key):
        self.val = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = Node(key)
        else:
            curr = self.root
            while True:
                if key < curr.val:
                    if not curr.left:
                        curr.left = Node(key)
                        break
                    else:
                        curr = curr.left
                else:
                    if not curr.right:
                        curr.right = Node(key)
                        break
                    else:
                        curr = curr.right

    def search(self, key):
        curr = self.root
        while curr and curr.val != key:
            if key < curr.val:
                curr = curr.left
            else:
                curr = curr.right
        return curr

def solution(lst, search_lst):
    answer = []

    bst = BST()

    for node in lst:
        bst.insert(node)

    for search in search_lst:
        if bst.search(search):
            answer.append(True)
        else:
            answer.append(False)

    return answer
print(solution(lst, search_lst))