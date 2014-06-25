
import collections

minKeyinTree = -10000000000000000
maxKeyinTree = 100000000000000000


class Node(object):
    def __init__(self, key, value, count, p=None, l=None, r=None):
        self.key = key
        self.value = value
        self.parent = None    # link to parent
        self.left = l         # link to left subtree
        self.right = r        # link to right subtree
        self.count = count

    def __repr__(self):
        return("Node({}, {}, {})".format(self.key, self.value, self.count))


class Tree(object):
    def __init__(self, key=None, value=None):
        if key:
            self.root = Node(key, value, 1)
        else:
            self.root = Node(None, None, 0)

    def get(self, key):
        """
        return value of key
        or None if not found
        """
        def helper(tree, key):
            if tree is None:
                return
            if key == tree.key:
                return tree.value  # could also return tree (node with key)
            elif key < tree.key:
                return helper(tree.left, key)
            else:
                return helper(tree.right, key)
        return helper(self.root, key)

    def put(self, key, value):
        def helper(tree, key, value):
            """
            insert key, value into tree or
            update value, if key found in tree
            return tree
            """
            if tree is None:
                # put node here
                return Node(key, value, 1, None, None, None)
            if key == tree.key:
                # update value
                tree.value = value
                return tree
            elif key < tree.key:
                tree.left = helper(tree.left, key, value)
                tree.left.parent = tree
            else:
                tree.right = helper(tree.right, key, value)
                tree.right.parent = tree
            # calculate count on the way up
            tree.count = self.size(tree.left) + self.size(tree.right) + 1
            return tree
        return helper(self.root, key, value)

    def minKey(self):
        """
        return min key
        """
        def helper(tree):
            if tree is None:
                return
            if tree.left:
                return helper(tree.left)
            else:
                return tree.key
        return helper(self.root)

    def minNode(self):
        """
        return node containing min key
        """
        def helper(tree):
            if tree is None:
                return
            if tree.left:
                return helper(tree.left)
            else:
                return tree
        return helper(self.root)

    def maxKey(self):
        """
        return max key
        """
        def helper(tree):
            if tree is None:
                return
            if tree.right:
                return helper(tree.right)
            else:
                return tree.key
        return helper(self.root)

    def maxNode(self):
        """
        return node containing max key
        """
        def helper(tree):
            if tree is None:
                return
            if tree.right:
                return helper(tree.right)
            else:
                return tree
        return helper(self.root)

    def floorKey(self, query):
        def helper(tree, query):
            """
            return node containing floor of query
            """
            if tree is None:
                return
            if tree.key == query:
                return tree
            elif query < tree.key:
                return helper(tree.left, query)
            else:
                t = helper(tree.right, query)
                if t:
                    return t
                else:
                    return tree
        return helper(self.root, query)

    def ceiling(self, query):
        def helper(tree, query):
            """
            return node containing ceiling of query
            """
            if tree is None:
                return
            if tree.key == query:
                return tree
            elif tree.key < query:
                return helper(tree.right, query)
            else:
                t = helper(tree.left, query)
                if t:
                    return t
                else:
                    return tree
        return helper(self.root, query)

    def size(self, t=None):
        """
        return number of keys rooted at this node
        """
        def helper(node):
            if node is None:
                return 0
            else:
                node.count = helper(node.left) + helper(node.right) + 1
                return node.count
        if t:
            return t.count
        else:
            return helper(self.root)

    def rank(self, key):
        def helper(tree, key):
            """
            return number of keys < key
            return rank (aka index of key)
            """
            if tree is None:
                return 0
            if key == tree.key:
                return self.size(tree.left)
            elif key < tree.key:
                return helper(tree.left, key)
            else:
                return self.size(tree.left) + 1 + helper(tree.right, key)
        return helper(self.root, key)

    def keys(self):
        """
        return a queue of keys
        inOrder: left, node, right
        """
        q = collections.deque()

        def helper(tree, lokey, hikey):
            if tree is None:
                return
            if lokey < tree.key:
                helper(tree.left, lokey, hikey)
            if lokey <= tree.key <= hikey:
                q.append(tree.key)
            if tree.key < hikey:
                helper(tree.right, lokey, hikey)
            return iter(q)
        return helper(self.root, self.minKey(), self.maxKey())

    def select(self, i):
        """
        return key with rank i

        aka: return key with index i
        """
        def helper(node, i):
            if node is None:
                return
            n = self.size(node.left)
            if n == i:
                return node.key
            elif n > i:
                return helper(node.left, i)
            else:
                return helper(node.right, i - n - 1)
        return helper(self.root, i)

    def deleteMin(self):
        """
        delete node with min key
        return root node
        """
        def helper(node):
            if node is None:
                return
            if node.left is None:
                return node.right
            else:
                node.left = helper(node.left)  # delete is done here
                node.count = self.size(node.left) + self.size(node.right) + 1
                return node
        return helper(self.root)

    def deleteMax(self):
        """
        delete node with max key
        return tree
        """
        def helper(node):
            if node is None:
                return
            if node.right is None:
                return node.left
            else:
                node.right = helper(node.right)  # delete is done here
                node.count = self.size(node.left) + self.size(node.right) + 1
                return node
        return helper(self.root)

    def succ(self, key):
        """
        returns the node that is the successor to the given key
        """
        def helper_min(node):
            if node is None:
                return
            if node.left:
                helper_min(node.left)
            else:
                return node

        def helper_succ(node, key):
            if node is None:
                return
            if key < node.key:
                return helper_succ(node.left, key)
            elif key > node.key:
                return helper_succ(node.right, key)
            else:
                # found node with key
                if node.right:
                    return helper_min(node.right)
                while node.parent and node.parent.key < key:
                    node = node.parent
                if node.parent:
                    return node.parent
                else:
                    return "no successor"
        return helper_succ(self.root, key)

    def pred(self, key):
        """
        returns the node that precedes given key
        """
        def helper_max(node):
            if node is None:
                return
            if node.right:
                return helper_max(node.right)
            else:
                return node

        def helper_pred(node, key):
            if node is None:
                return
            if key < node.key:
                return helper_pred(node.left, key)
            elif key > node.key:
                return helper_pred(node.right, key)
            else:
                # found given node
                if node.left:
                    # return maxKey of left subtree
                    return helper_max(node.left)
                else:
                    while node.parent and node.parent.key > key:
                        node = node.parent
                    # found predecessor node
                    if node.parent:
                        return node.parent
                    else:
                        return "no predecessor"
        return helper_pred(self.root, key)


def isBST(tree):
    """return True if input tree is binary search tree"""
    def helper(tree, minkey, maxkey):
        if tree is None:
            return True
        if tree.key < minkey or tree.key > maxkey:
            return False
        elif (not helper(tree.left, minkey, tree.key) or
                not helper(tree.right, tree.key, maxkey)):
            return False
        else:
            return True
    # need a min that is <= all keys in tree
    # need a max that is >= all keys in tree
    return helper(tree, minKeyinTree, maxKeyinTree)


def dfs_order(tree):
    def helper(node):
        if not node:
            return
        dfs_order(node.left)
        dfs_order(node.right)
        print(node.key)
    return helper(tree.root)


def dfs_order_g(tree):
    def helper(node):
        if node:
            for x in dfs_order_g(node.left):
                yield x
            for x in dfs_order_g(node.right):
                yield x
            yield node.key
    return helper(tree.root)


def printTreeBFS(tree):
    q = collections.deque()
    if tree:
        q.append(tree)
    while len(q) > 0:
        t = q.popleft()
        print(t.key)
        if t.left:
            q.append(t.left)
        if t.right:
            q.append(t.right)


def bfs_order_g(tree):
    q = collections.deque()
    if tree:
        q.append(tree)
    while len(q) > 0:
        t = q.popleft()
        yield t.value
        if t.left:
            q.append(t.left)
        if t.right:
            q.append(t.right)


def printTreeInOrder(tree):
    """left, value, right"""
    def helper(node):
        if node is None:
            return
        else:
            printTreeInOrder(node.left)
            print(node.value, end=' ')
            printTreeInOrder(node.right)
    return helper(tree.root)


def inOrder(tree):
    def helper(node):
        if node:
            for x in inOrder(node.left):
                yield x
            yield node.value
            for x in inOrder(node.right):
                yield x
    return helper(tree.root)


def preOrder(tree):
    def helper(node):
        if node:
            yield node.value
            for x in preOrder(node.left):
                yield x
            for x in preOrder(node.right):
                yield x
    return helper(tree.root)

####################################
if __name__ == "__main__":
    tree = Tree(20, "A")
    tree.put(18, "B")
    tree.put(25, "E")
    tree.put(17, "C")
    tree.put(19, "D")
    tree.put(22, "F")
    tree.put(26, "G")
for k in tree.keys():
    print(k)
p18 = tree.pred(18)
p19 = tree.pred(19)
p22 = tree.pred(22)
print('18:', p18.key, '19:', p19.key, '22:', p22.key)
s18 = tree.succ(18)
s19 = tree.succ(19)
s26 = tree.succ(26)
print(s18, s19, s26)
m = tree.deleteMax()
m = tree.deleteMin()


def delete(tree, key):
    """
    delete key from tree
    return tree
    """
    if tree is None:
        return
    if key < tree.key:
        tree.left = delete(tree.left, key)
    elif key > tree.key:
        tree.right = delete(tree.right, key)
    else:
        # found node to delete
        if tree.right is None:
            return tree.left
        if tree.left is None:
            return tree.right
        else:
            # node to delete has 2 children
            t = tree
            x = tree.succ(tree.key)
            x.right = tree.right.deleteMin()
            x.left = t.left
    x.count = x.size(x.left) + x.size(x.right) + 1
    return x


def isCountCorrect(tree):
    def helper(node):
        if node is None:
            return True
        if node.left is None and node.right is None:
            return node.count == 1
        return isCountCorrect(node.left) and isCountCorrect(node.right) and \
            node.count == node.left.count + node.right.count + 1
    return helper(tree.root)


def isSelectRankCorrect(tree):
    N = tree.size()
    for i in range(N):
        if tree.rank(tree.select(i)) != i:
            return False
    for k in tree.keys():
        if tree.select(tree.rank(k)) != k:
            return False
    return True


def height(tree):
    """
    returns the height of tree
    """
    def helper(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 0
        return 1 + max(height(node.left), height(node.right))
    return helper(tree.root)


def printLevel(tree):
    """
    print keys in level order
    left -> right, for each level of tree
    """
    if tree is None:
        return
    q = collections.deque()
    q.append(tree)
    nodes_in_current_level = 1
    nodes_in_next_level = 0
    while len(q):
        t = q.popleft()
        nodes_in_current_level -= 1
        if t is None:
            continue
        q.append(t.left)
        q.append(t.right)
        nodes_in_next_level += 2
        if nodes_in_current_level > 0:
            print(t.key)
        else:
            print(t.key)
            nodes_in_current_level = nodes_in_next_level
            nodes_in_next_level = 0


def avgCompares(tree):
    """
    return avgCompares for a successfull search
    1 + (internal path length / N)
    """
    N = tree.size()
    return 1 + internalPathLength(tree) / N


def isLeaf(tree):
    def helper(node):
        if node is None:
            return False
        if node.left is None and node.right is None:
            return True
        else:
            return False
    return helper(tree.root)


def internalPathLength(tree):
    """
    return total distance of all internal nodes from the root
    """
    if tree is None:
        return 0
    if isLeaf(tree):
        return 0
    return (internalPathLength(tree.root.left) +
            internalPathLength(tree.root.right)
            + tree.size() - 1)

"""

        0
       / \
      /   \
     1     2
    / \   / \
   3   4 5   6
  /   / \   /
 7   8   9 10

interface INode {
  INode getLeft();
  INode getRight();
  INode getParent();
}

(7,4) => 1
(8,4) => 4

"""
"""
Trees below here, are not objects of the Tree class.
They are just concept trees with a left and right subtree.
"""


def nearestAncestor(INodeA, INodeB):
    """find nearest ancestor INodeB"""
    nodeA = INodeA
    nodeB = INodeB
    countA = 0
    countB = 0

    if nodeA == nodeB:
        return nodeA

    # find depth in tree
    while nodeA.parent is not None:
        nodeA = nodeA.getParent()
        countA = countA + 1
    while nodeB.parent is not None:
        nodeB = nodeB.getParent()
        countB = countB + 1

    # find difference in depths
    diff = countA - countB
    if diff > 0:
        for _ in range(diff):
            nodeA = nodeA.getParent()
    elif diff < 0:
        for _ in range(diff):
            nodeB = nodeB.getParent()

    # nearest ancestor = first node where nodeA and nodeB are equal
    while nodeB != nodeA:
        nodeB = nodeB.getParent()
        nodeA = nodeA.getParent()

    # return nearest ancestor
    return nodeB


def find_paths_sum_to_N(tree, n):
    """
    You are given a binary tree in which each node contains a value.
    Design an algorithm to print all paths which sum to a given value.
    The path does not need to start or end at the root or a leaf.
    """
    list_of_paths = find_all_paths(tree)
    print(filter(lambda x: sum(x) == n, list_of_paths))
    return


def find_all_paths(tree):
    """
    return a list of paths
    each path is represented as a sequence of keys
    """
    if tree is None:
        return []
    if isLeaf(tree):
        return [[tree.key]]
    a = find_all_paths(tree.left)
    b = find_all_paths(tree.right)
    s = [x for x in a if tree.left.key in x]
    a = a + map(lambda x: x + [tree.key], s)
    s = [x for x in b if tree.right.key in x]
    b = b + map(lambda x: x + [tree.key], s)
    return a + b + [[tree.key]]


def DFSPaths(G, s, t):
    """
    Given a directed graph,
    design an algorithm to find out whether there is a route between two nodes.

    Use DFS.
    """
    marked = [False] * G.V()
    marked[s] = True
    marked = dfs(G, s, marked)
    return marked[t]


def dfs(G, v, marked):
    marked[v] = True
    for w in G.adj(v):
        if not marked[w]:
            marked = dfs(G, w, marked)
    return marked


def linkNodes(tree):
    """
    Given a binary search tree, design an algorithm which creates
    a linked list of all the nodes at each depth (i.e., if you have
    a tree with depth D, you will have D linked lists)
    """
    if tree is None:
        return
    q = collections.deque()
    q.append(tree)
    nodes_in_current_level = 1
    nodes_in_next_level = 0

    list_of_linked_lists = [None] * height(tree)
    level = 0
    while len(q):
        t = q.popleft()
        nodes_in_current_level -= 1
        if t is None:
            continue
        q.append(t.left)
        q.append(t.right)
        nodes_in_next_level += 2

        if level > 0:
            if list_of_linked_lists[level - 1] is None:
                # start new list
                list_of_linked_lists[level - 1] = list(Node(t.key))
            else:
                list_of_linked_lists[level - 1] = \
                    list_of_linked_lists[level - 1].appendToTail(t.key)

        if nodes_in_current_level > 0:
            print(t.key, end=' ')
        else:
            print(t.key)
            nodes_in_current_level = nodes_in_next_level
            nodes_in_next_level = 0

            # start new level
            level += 1
    return list_of_linked_lists


# a = linkNodes(tree1)  # a is a list of linked lists
# g = (n for n in a)
# for n in g:
#     print n


def isSubTree(t2, t1):
    """
    You have two very large binary trees: T1, with millions of nodes,
    and T2, with hundreds of nodes. Create an algorithm to decide if
    T2 is a subtree of T1.
    """
    if t2 is None and t1 is None:
        return True
    if t1 is None or t2 is None:
        return False
    if t2.key == t1.key:
        # check that all keys in t2 match keys in t1
        return [1 for i, j in zip(t2.keys(), t1.keys()) if i != j] == []
    elif t2.key < t1.key:
        return isSubTree(t2, t1.left)
    else:
        return isSubTree(t2, t1.right)


def isBalanced(tree):
    """
    Implement a function to check if a tree is balanced. For this
    question, a balanced tree is defined to be a tree such that no 2 leaf nodes
    differ in distance from the root by more than one.
    """
    if tree is None:
        return True
    if isLeaf(tree):
        return True
    return (isBalanced(tree.left) and isBalanced(tree.right) and
            abs(height(tree.left) - height(tree.right)) < 2)


def createBST(sorted_a):
    """
    Given a sorted (increasing order) array of keys, write an
    algorithm to create a binary search tree with minimal height.
    """
    a = list(sorted_a)       # shallow copy
    N = len(a)
    if N == 0:
        return
    if N == 1:
        return Node(a[0], 0, 1, None, None)
    mid = N // 2
    root = Node(a[mid], mid, 1, None, None)
    root.left = createBST(root.left, a[0:mid])
    root.right = createBST(root.right, a[mid + 1:])
    return root


def createBTree(a):
    """
    create a binary tree (not BST) from an unordered array
    array starts at 0
    for array index i:
    root: i
    left = 2*i + 1
    right = 2*i + 2
    """
    q = collections.deque()
    root = Node(a[0], a[0], 1, None, None)
    q.append(root)
    i = 0
    while len(q) > 0:
        tree = q.popleft()
        left = 2 * i + 1
        if left >= len(a):
            break
        tree.left = Node(a[left], a[left], 1, None, None)
        q.append(tree.left)
        right = 2 * i + 2
        if right >= len(a):
            break
        tree.right = Node(a[right], a[right], 1, None, None)
        q.append(tree.right)
        i += 1
    return root


def treemap(proc, tree):
    """
    maps a procedure over a tree structure
    """
    res = []
    if tree is None:
        return res
    if isLeaf(tree):
        return [proc(tree.key)]
    else:
        return [proc(tree.key)] + treemap(proc, tree.left) + \
            treemap(proc, tree.right)


def treemap2(proc, nested_list):
    """
    map proc to a nested list

    """
    res = []
    if nested_list is None:
        return
    if nested_list is []:
        return res
    else:
        for x in nested_list:
            if isinstance(x, list):
                res = res + treemap2(proc, x)
            else:
                res = res + [proc(x)]
        return res


def list_reverse(nested_list):
    """
    reverse a nested list
    """
    def reverse_list(alist):
        if alist is None:
            return
        if alist is []:
            return []
        if len(alist) == 1:
            return alist
        rev = reverse_list(alist[1:])
        rev.append(alist[0])
        return rev
    return treemap2(reverse_list, nested_list)


