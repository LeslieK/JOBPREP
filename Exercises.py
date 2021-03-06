def fib(n):
	'''find fibonacci number of n'''
	def helper(a, b, count):
		if count == 0:
			return b
		else:
			return helper(a + b, a, count - 1)
	return helper(1, 0, n)


def sum(list):
	'''add the integers in a list'''
	num_elements = len(list)

	def sum_iter(acc, count):
		'''sum the itegers in a list'''
		if list == []:
			return 0
		if count > 0:
			return sum_iter(acc + list[count - 1], count - 1)
		else:
			return acc                 
	return sum_iter(0, num_elements)

def lastIndexOf(n, list):
	'''return the last index of n in a list'''
	if list == []:
		return -1
	def helper(index, i, count):
		if i < count:
			if list[i] == n:
				index = i
			return helper(index, i + 1, count)
		else:
			return index
	return helper(-1, 0, len(list))

#lastIndexOf(5, [1,2,3,4,5,6,7,5,8,9,5,4])

# write a recursive function to compute sum of numbers stored in a binary tree
class Node(object):
	def __init__(self, value, l=None, r=None):
		self.value = value
		self.left = l
		self.right = r

tree1 = Node(1, Node(2, Node(3), Node(4)), Node(5, Node(6), Node(7)))
tree2 = Node(1, Node(2, Node(3)), Node(2))
tree3 = Node(1)

def sumTree(tree):
	'''sum of integer elements in tree'''
	root = tree.value
	left = tree.left
	right = tree.right
	def helper(acc, tree, parents):
		#print 'len parents: {}, root: {}'.format(len(parents), parents[-1].value)
		if tree is None:
			return acc
		parents.append(tree)
		acc = helper(acc + tree.value, tree.left, parents)
		acc = helper(acc, tree.right, parents)
		parents.pop()
		return acc
	return helper(0, tree, [tree])

def printTreeDFS(tree):
	'''prints elements in DFS order'''
	root = tree.value
	left = tree.left
	right = tree.right
	stack = []
	def dfs(tree, stack):
		if tree is None:
			return stack
		stack = dfs(tree.left, stack)
		stack = dfs(tree.right, stack)
		stack.append(tree.value)
		return stack
	print dfs(tree, stack)

import collections
def printTreeBFS(tree):
	q = collections.deque()
	q.append(tree)
	while len(q) > 0:
		tree = q.popleft()
		if tree is not None:
			print tree.value
			q.append(tree.left)
			q.append(tree.right)


def numberOfWays(n, cache):
	'''number of ways to climb a staircase of n steps

	can climb 1, 2, or 3 steps at a time'''

	if n < 1:
		return 0
	elif n == 1:
		return 1
	elif n == 2:
		return 2
	elif n == 3:
		return 4
	else:
		key = 'numberOfWays(' + str(n) + ')'
		if key in cache:
			return cache[key]
		else:
			cache[key] = numberOfWays(n - 1, cache) + numberOfWays(n - 2, cache) + numberOfWays(n - 3, cache)
		return cache[key]

def numberOfWaysTaxi(row, col, cache):
	'''number of ways to get from start to goal

	NxN grid; only move right or down'''
	if row == 0 and col == 0:
		return 0
	elif row == 0 and col == 1:
		return 1
	elif row == 1 and col == 0:
		return 1
	else:
		key = (row, col)
		if key in cache:
			return cache[key]
		elif col == 0:
			# on left border
			cache[key] = numberOfWaysTaxi(row-1, col, cache)
		elif row == 0:
			# on top border
			cache[key] = numberOfWaysTaxi(row, col-1, cache)
		else:
			cache[key] = numberOfWaysTaxi(row-1, col, cache) + numberOfWaysTaxi(row, col-1, cache)
		print cache[key]
		return cache[key]
#numberOfWaysTaxi(10, 10, {})

def nChooseKCount(n, k, cache):
	'''compute the number of ways of choosing k elements from a set of n elements'''

	if k == n:
		return 1
	elif k == 1:
		return n
	elif k == 0:
		return 1
	else:
		key = (n, k)
		if key in cache:
			return cache[key]
		else:
			cache[key] = nChooseKCount(n-1, k-1, cache) + nChooseKCount(n-1, k, cache)
		return cache[key]


# a set is a list of unique elements (use 'list' not 'set')
def subsets(s):
	'''compute list of subsets from a list of unique items'''
	if s == []:
		return [[]]
	else:
		rest = subsets(s[1:])
		return rest + map(lambda x: ([s[0]] + x), rest)

def subsets_k(s, k):
	def subsets(s):
		'''compute list of subsets from a list of unique items'''
		if s == []:
			return [[]]
		else:
			rest = subsets(s[1:])
			return rest + map(lambda x: ([s[0]] + x), rest)
	r = subsets(s)
	return filter(lambda x: len(x) == k, r)


def sublists(big_list, selected_so_far):
	if big_list == []:
		return [selected_so_far]
	else:
		curr_element = big_list[0]
		rest_of_big_list = big_list[1:]
		return (sublists(rest_of_big_list, selected_so_far) + 
			sublists(rest_of_big_list, selected_so_far + [curr_element]))
