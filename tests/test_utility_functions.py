import pytest
import numpy as np


def closest(array, value):
    n = len(array)
    if hasattr(value, '__len__') == False:
        if (value < array[0]):
            return 0
        elif (value > array[n-1]):
            return n-1
        jl = 0
        ju = n-1
        while (ju-jl > 1):
            jm = (ju+jl) >> 1
            if (value >= array[jm]):
                jl = jm
            else:
                ju = jm
        if (value == array[0]):
            return 0
        elif (value == array[n-1]):
            return n-1
        else:
            jn = jl + np.argmin([value-array[jl], array[jl+1]-value])
            return jn
    else:
        nv = len(value)
        jn = np.zeros(nv, dtype='int32')
        for i in range(nv):
            if (value[i] < array[0]): 
                jn[i] = 0
            elif (value[i] > array[n-1]): 
                jn[i] = n-1
            else:
                jl = 0
                ju = n-1
                while (ju-jl > 1):
                    jm=(ju+jl) >> 1
                    if (value[i] >= array[jm]):
                        jl = jm
                    else:
                        ju = jm
                if (value[i] == array[0]):
                    jn[i] = 0
                elif (value[i] == array[n-1]):
                    jn[i] = n-1
                else:
                    jn[i] = jl+np.argmin([value[i]-array[jl], array[jl+1]-value[i]])
        return jn


def where_v(elements, array, approx=False, assume_sorted=True):
    if isinstance(array, list): array = np.array(array)
    try:
        dd = len(elements)
        if isinstance(elements, list): 
            elements = np.array(elements)
        dim = len(elements.shape)
    except TypeError: dim = 0

    if approx == True:
        if assume_sorted == False:
            i_sort = np.argsort(array)
            array2 = array[i_sort]
        else:
            array2 = array
        if dim == 0:
            w = closest(array2, elements)
            return w
        ind = np.zeros(len(elements), dtype=np.int16)
        for i in range(len(elements)):
            ind[i] = closest(array2,elements[i])
        if assume_sorted:
            return ind
        else:
            return i_sort[ind]
    else:
        if dim == 0:
            w, = np.where(array == elements)
            return w
        ind = np.zeros(len(elements), dtype=np.int16)
        for i in range(len(elements)):
            w, = np.where(array == elements[i])
            if len(w) == 0: 
                ind[i] = len(array)
            else: 
                ind[i] = w[0]
                
        return ind


def nansumwrapper(a, axis=None, **kwargs):
    ma = np.isnan(a) == False
    sa = np.nansum(ma, axis=axis)
    sm = np.nansum(a, axis=axis,**kwargs)
    sm = np.where(sa == 0, np.nan, sm)
    return sm


def add_mag(mag1, mag2):
    return -2.5*np.log10(10**(-0.4*mag1)+10**(-0.4*mag2))


class TestClosest:
    def test_closest_exact_match(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert closest(arr, 3.0) == 2

    def test_closest_single_value_above_range(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert closest(arr, 10.0) == 4

    def test_closest_single_value_below_range(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert closest(arr, 0.0) == 0

    def test_closest_single_value_midpoint(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = closest(arr, 3.5)
        assert result == 2

    def test_closest_list_of_values(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        values = [3.5, 5.0, 10.0, 0.5]
        result = closest(arr, values)
        expected = np.array([2, 4, 9, 0])
        np.testing.assert_array_equal(result, expected)

    def test_closest_edge_cases(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert closest(arr, 1.0) == 0
        assert closest(arr, 3.0) == 2


class TestWhereV:
    def test_where_v_exact_match(self):
        arr = np.array([10, 20, 30, 40, 50])
        elements = [20, 40]
        result = where_v(elements, arr)
        np.testing.assert_array_equal(result, [1, 3])

    def test_where_v_no_match(self):
        arr = np.array([10, 20, 30, 40, 50])
        elements = [25, 35]
        result = where_v(elements, arr)
        np.testing.assert_array_equal(result, [5, 5])

    def test_where_v_approx_mode(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        elements = [2.3, 4.7]
        result = where_v(elements, arr, approx=True)
        np.testing.assert_array_equal(result, [1, 4])

    def test_where_v_approx_unsorted(self):
        arr = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        elements = [3.0, 7.0]
        result = where_v(elements, arr, approx=True, assume_sorted=False)
        np.testing.assert_array_equal(result, [1, 2])

    def test_where_v_single_element(self):
        arr = np.array([10, 20, 30])
        result = where_v(20, arr)
        np.testing.assert_array_equal(result, [1])


class TestNansumwrapper:
    def test_nansumwrapper_normal(self):
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        result = nansumwrapper(arr)
        assert result == 7.0

    def test_nansumwrapper_all_nans(self):
        arr = np.array([np.nan, np.nan, np.nan])
        result = nansumwrapper(arr)
        assert np.isnan(result)

    def test_nansumwrapper_axis_0(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = nansumwrapper(arr, axis=0)
        np.testing.assert_array_equal(result, [4.0, 4.0])

    def test_nansumwrapper_axis_1(self):
        arr = np.array([[1.0, 2.0, np.nan], [np.nan, 4.0, 5.0]])
        result = nansumwrapper(arr, axis=1)
        np.testing.assert_array_equal(result, [3.0, 9.0])

    def test_nansumwrapper_empty_array(self):
        arr = np.array([])
        result = nansumwrapper(arr)
        assert np.isnan(result)


class TestAddMag:
    def test_add_mag_basic(self):
        result = add_mag(1.0, 2.0)
        assert np.isclose(result, 0.636, rtol=1e-3)

    def test_add_mag_equal_magnitudes(self):
        result = add_mag(5.0, 5.0)
        assert np.isclose(result, 4.247, rtol=1e-3)

    def test_add_mag_different_magnitudes(self):
        result = add_mag(10.0, 15.0)
        assert np.isclose(result, 10.004, rtol=1e-2)

    def test_add_mag_zero(self):
        result = add_mag(0.0, 0.0)
        expected = -2.5 * np.log10(2)
        assert np.isclose(result, expected)

    def test_add_mag_symmetry(self):
        assert np.isclose(add_mag(3.0, 5.0), add_mag(5.0, 3.0))