# distutils: language = c++
# cython: language_level = 3
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

def is_float_32():
    cdef size_of_float = sizeof(float)*8
    return size_of_float==32

def is_int_32():
    cdef size_of_int = sizeof(int)*8
    return size_of_int==32


assert is_int_32()
assert is_float_32()
