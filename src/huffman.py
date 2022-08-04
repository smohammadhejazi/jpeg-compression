from queue import PriorityQueue


# calculate frequencies
def calculate_freq(arr):
    frequencies = dict()
    for i in arr:
        if i in frequencies:
            frequencies[i] += 1
        else:
            frequencies[i] = 1
    return frequencies


class Node:
    def __init__(self, value=None, freq=None, left_child=None, right_child=None):
        self.value = value
        self.freq = freq
        self.left_child = left_child
        self.right_child = right_child

    # initialize a leaf node
    def init_leaf(self, value, freq):
        return Node(value, freq, None, None)

    # initialize a non-leaf node
    def init_node(self, left_child, right_child):
        freq = left_child.freq + right_child.freq
        return Node(None, freq, left_child, right_child)

    def is_leaf(self):
        return self.value is not None

    # define ==, !=, <, <=, >, >=
    def __eq__(self, other):
        if self.value == other.value and \
                self.freq == other.freq and \
                self.left_child == other.left_child and\
                self.right_child == other.right_child:
            return True
        return False

    def __nq__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.freq < other.freq

    def __le__(self, other):
        return self.freq < other.freq or self.freq == other.freq

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)


class HuffmanTree:
    def __init__(self, arr):
        self.value_to_bitstring_table = dict()
        q = PriorityQueue()

        for val, freq in calculate_freq(arr).items():
            node = Node().init_leaf(value=val, freq=freq)
            q.put(node)

        while q.qsize() >= 2:
            left = q.get()
            right = q.get()
            node = Node().init_node(left_child=left, right_child=right)
            q.put(node)

        self.root = q.get()
        self.create_huffman_table()

    # build the table
    def create_huffman_table(self):
        def tree_traverse(current_node, bits=''):
            if current_node is None:
                return
            if current_node.is_leaf():
                self.value_to_bitstring_table[current_node.value] = bits.encode()
                return
            tree_traverse(current_node.left_child, bits + '0')
            tree_traverse(current_node.right_child, bits + '1')

        tree_traverse(self.root)

    # get value to bitstring table
    def get_value_to_bitstring_table(self):
        return self.value_to_bitstring_table
