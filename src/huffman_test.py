from huffman import *

if __name__ == '__main__':
    array = [8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10]
    tree = HuffmanTree(array)
    table = tree.get_value_to_bitstring_table()
    bits = bytes()
    for i in array:
        bits += (table[i])

    print(bits)
    print(len(bits))
