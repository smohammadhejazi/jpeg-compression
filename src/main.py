import numpy as np
from PIL import Image
from scipy.fftpack import dct
from huffman import *
import pickle
import math


PHOTO_PATH = "./photo.png"
OUTPUT_FILE = "./photo_encoded.txt"
YCBCR_SUBSAMPLE_PHOTO = "./photo_ycbcr_subsample.jpeg"


QUANTIZATION_MATRICES = {
    "luminance": np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                           [12, 12, 14, 19, 26, 58, 60, 55],
                           [14, 13, 16, 24, 40, 57, 69, 56],
                           [14, 17, 22, 29, 51, 87, 80, 62],
                           [18, 22, 37, 56, 68, 109, 103, 77],
                           [24, 35, 55, 64, 81, 104, 113, 92],
                           [49, 64, 78, 87, 103, 121, 120, 101],
                           [72, 92, 95, 98, 112, 100, 103, 99]]),

    "chrominance": np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                             [18, 21, 26, 66, 99, 99, 99, 99],
                             [24, 26, 56, 99, 99, 99, 99, 99],
                             [47, 66, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99, 99, 99]])
}


def load_image():
    image_file = Image.open(PHOTO_PATH)
    image_file = image_file.convert('YCbCr')
    return np.array(image_file, dtype=np.uint8)


def save_image_as(image, path):
    image = Image.fromarray(image, 'YCbCr')
    image.save(path)


def subsampling_420(image):
    new_image = image.copy()
    # row subsampling
    if new_image.shape[0] % 2 == 0:
        new_image[1::2, :, 1:3] = new_image[::2, :, 1:3]
    else:
        last_row = new_image.shape[0] - 2
        new_image[1::2, :, 1:3] = new_image[:last_row:2, :, 1:3]

    # column subsampling
    if new_image.shape[1] % 2 == 0:
        new_image[:, 1::2, 1:3] = new_image[:, ::2, 1:3]
    else:
        last_col = new_image.shape[1] - 2
        new_image[:, 1::2, 1:3] = new_image[:, :last_col:2, 1:3]

    return new_image
    # return np.sum(new_image, axis=2)


def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def quantize(dct_block):
    dct_block[:, :, 0] /= QUANTIZATION_MATRICES['luminance']
    dct_block[:, :, 1] /= QUANTIZATION_MATRICES['chrominance']
    dct_block[:, :, 2] /= QUANTIZATION_MATRICES['chrominance']
    return dct_block


def block_to_zigzag(block):
    zigzag = np.empty(shape=(block.shape[0] * block.shape[1], block.shape[2]))
    for k in range(3):
        zigzag[:, k] = np.concatenate([np.diagonal(block[::-1, :, k], i)[::(2 * (i % 2) - 1)]
                                       for i in range(1 - block.shape[0], block.shape[0])])[:]
    return zigzag


def get_dc_ac(image, width):
    # if image cannot be divided to width * width blocks
    if image.shape[0] % 4 != 0:
        short = math.ceil(image.shape[0] / 4) * 4 - image.shape[0]
        image = np.append(image, np.zeros(shape=(short, image.shape[1], 3)), axis=0)

    if image.shape[1] % 4 != 0:
        short = math.ceil(image.shape[1] / 4) * 4 - image.shape[1]
        image = np.append(image, np.zeros(shape=(image.shape[0], short, 3)), axis=1)

    blocks_count = (image.shape[0] // width) * (image.shape[1] // width)

    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, (width * width - 1), 3), dtype=np.int32)

    block_index = 0
    num_elements = width * width
    for i in range(0, image.shape[0], width):
        for j in range(0, image.shape[1], width):
            block = image[i:i+width, j:j+width, :]
            dct_block = dct_2d(block)
            quantized_block = quantize(dct_block)
            zigzag_block = block_to_zigzag(quantized_block)
            dc[block_index, :] = zigzag_block[0, :]
            ac[block_index, :, :] = zigzag_block[1:num_elements, :]
            block_index += 1

    return dc, ac


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)


def binstr_flip(binstr):
    # check if binstr is a binary string
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    # find last non zero
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    # symbol = (run_length, value)
    symbols = []
    run_length = 0
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            symbols.append((run_length, elem))
            run_length = 0
    return symbols


def save_to_file(filepath, dc, ac, blocks_count, tables):
    with open(filepath, 'wb') as file:
        # save huffman tables
        pickle.dump(tables, file, pickle.HIGHEST_PROTOCOL)

        # save encoded ac, dc
        for b in range(blocks_count):
            for c in range(3):
                dc_table = {}
                if c == 0:
                    dc_table = tables['dc_y']
                else:
                    dc_table = tables['dc_c']

                ac_table = {}
                if c == 0:
                    ac_table = tables['ac_y']
                else:
                    ac_table = tables['ac_c']

                # save dc
                file.write(dc_table[dc[b, c]])
                # save ac
                symbols = run_length_encode(ac[b, :, c])
                for i in range(len(symbols)):
                    file.write(ac_table[tuple(symbols[i])])


def dpcm(array):
    array[1:] = array[1:] - array[0]
    return array


if __name__ == '__main__':
    print("Compressing {} to {}".format(PHOTO_PATH, OUTPUT_FILE))
    image = load_image()
    image = subsampling_420(image)
    save_image_as(image, YCBCR_SUBSAMPLE_PHOTO)
    print("YCbCr photo path: {}".format(YCBCR_SUBSAMPLE_PHOTO))
    dc, ac = get_dc_ac(image, 8)
    block_count = ac.shape[0]
    huffman_dc_y = HuffmanTree(dpcm(dc[:, 0]))
    huffman_dc_c = HuffmanTree(dpcm(dc[:, 1:].flat))
    huffman_ac_y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0]) for i in range(block_count)))
    huffman_ac_c = HuffmanTree(flatten(run_length_encode(ac[i, :, j]) for i in range(block_count) for j in [1, 2]))
    tables = {'dc_y': huffman_dc_y.get_value_to_bitstring_table(),
              'dc_c': huffman_dc_c.get_value_to_bitstring_table(),
              'ac_y': huffman_ac_y.get_value_to_bitstring_table(),
              'ac_c': huffman_ac_c.get_value_to_bitstring_table()}
    save_to_file(OUTPUT_FILE, dc, ac, block_count, tables)
    print("Done.")
