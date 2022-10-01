import random
import string


def chunker(length, size):
    return ((start, start + size) for start in range(0, length, size))


def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
