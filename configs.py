# coding: utf8

_letter_cases = 'abdefghmnpqrstwxyz'  # 小写字母，去除可能干扰的c i j k l o u v 18
_upper_cases = 'ABDEFHMNPQRSTWXYZ'  # 大写字母，去除可能干扰的C G I J K L O U V 17
_numbers = ''.join(map(str, range(2, 10)))  # 数字，去除0，1 (8)
INIT_CHARS = ''.join((_letter_cases, _upper_cases, _numbers))
CHARS = dict(zip(INIT_CHARS, range(len(INIT_CHARS))))

NUM_OF_LABELS = 6
NUM_OF_CLASSES = len(CHARS)


WIDTH = 96
HEIGHT = 25
NUM_CHANNELS = 3

