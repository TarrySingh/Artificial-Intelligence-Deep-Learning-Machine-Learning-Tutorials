
# NumPy: creating and manipulating numerical data 

## What is NumPy and numpy arrays

**Python:** - has built-in
- containers (costless insertion and append), dictionnaries (fast lookup)
- high-level number objects (integers, floating points)

**NumPy** is:
- extension package to Python to multidimensional arrays
- faster (as you'll see below)
- convenient and tested by scientific community


```python
import numpy as np
```


```python
a = np.array([0,1,2,3])
a
```




    array([0, 1, 2, 3])




```python
l = range(1000)

%timeit [i**2 for i in l]
```

    353 µs ± 2.58 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
a = np.arange(1000)
%timeit a**2
```

    1.52 µs ± 15.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
help(np.array)
```

    Help on built-in function array in module numpy.core.multiarray:
    
    array(...)
        array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
        
        Create an array.
        
        Parameters
        ----------
        object : array_like
            An array, any object exposing the array interface, an object whose
            __array__ method returns an array, or any (nested) sequence.
        dtype : data-type, optional
            The desired data-type for the array.  If not given, then the type will
            be determined as the minimum type required to hold the objects in the
            sequence.  This argument can only be used to 'upcast' the array.  For
            downcasting, use the .astype(t) method.
        copy : bool, optional
            If true (default), then the object is copied.  Otherwise, a copy will
            only be made if __array__ returns a copy, if obj is a nested sequence,
            or if a copy is needed to satisfy any of the other requirements
            (`dtype`, `order`, etc.).
        order : {'K', 'A', 'C', 'F'}, optional
            Specify the memory layout of the array. If object is not an array, the
            newly created array will be in C order (row major) unless 'F' is
            specified, in which case it will be in Fortran order (column major).
            If object is an array the following holds.
        
            ===== ========= ===================================================
            order  no copy                     copy=True
            ===== ========= ===================================================
            'K'   unchanged F & C order preserved, otherwise most similar order
            'A'   unchanged F order if input is F and not C, otherwise C order
            'C'   C order   C order
            'F'   F order   F order
            ===== ========= ===================================================
        
            When ``copy=False`` and a copy is made for other reasons, the result is
            the same as if ``copy=True``, with some exceptions for `A`, see the
            Notes section. The default order is 'K'.
        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise
            the returned array will be forced to be a base-class array (default).
        ndmin : int, optional
            Specifies the minimum number of dimensions that the resulting
            array should have.  Ones will be pre-pended to the shape as
            needed to meet this requirement.
        
        Returns
        -------
        out : ndarray
            An array object satisfying the specified requirements.
        
        See Also
        --------
        empty, empty_like, zeros, zeros_like, ones, ones_like, full, full_like
        
        Notes
        -----
        When order is 'A' and `object` is an array in neither 'C' nor 'F' order,
        and a copy is forced by a change in dtype, then the order of the result is
        not necessarily 'C' as expected. This is likely a bug.
        
        Examples
        --------
        >>> np.array([1, 2, 3])
        array([1, 2, 3])
        
        Upcasting:
        
        >>> np.array([1, 2, 3.0])
        array([ 1.,  2.,  3.])
        
        More than one dimension:
        
        >>> np.array([[1, 2], [3, 4]])
        array([[1, 2],
               [3, 4]])
        
        Minimum dimensions 2:
        
        >>> np.array([1, 2, 3], ndmin=2)
        array([[1, 2, 3]])
        
        Type provided:
        
        >>> np.array([1, 2, 3], dtype=complex)
        array([ 1.+0.j,  2.+0.j,  3.+0.j])
        
        Data-type consisting of more than one element:
        
        >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
        >>> x['a']
        array([1, 3])
        
        Creating an array from sub-classes:
        
        >>> np.array(np.mat('1 2; 3 4'))
        array([[1, 2],
               [3, 4]])
        
        >>> np.array(np.mat('1 2; 3 4'), subok=True)
        matrix([[1, 2],
                [3, 4]])
    



```python
np.lookfor('create array')
```

    Search results for 'create array'
    ---------------------------------
    numpy.array
        Create an array.
    numpy.memmap
        Create a memory-map to an array stored in a *binary* file on disk.
    numpy.diagflat
        Create a two-dimensional array with the flattened input as a diagonal.
    numpy.fromiter
        Create a new 1-dimensional array from an iterable object.
    numpy.partition
        Return a partitioned copy of an array.
    numpy.ctypeslib.as_array
        Create a numpy array from a ctypes array or a ctypes POINTER.
    numpy.ma.diagflat
        Create a two-dimensional array with the flattened input as a diagonal.
    numpy.ma.make_mask
        Create a boolean mask from an array.
    numpy.ctypeslib.as_ctypes
        Create and return a ctypes object from a numpy array.  Actually
    numpy.ma.mrecords.fromarrays
        Creates a mrecarray from a (flat) list of masked arrays.
    numpy.ma.mvoid.__new__
        Create a new masked array from scratch.
    numpy.lib.format.open_memmap
        Open a .npy file as a memory-mapped array.
    numpy.ma.MaskedArray.__new__
        Create a new masked array from scratch.
    numpy.lib.arrayterator.Arrayterator
        Buffered iterator for big arrays.
    numpy.ma.mrecords.fromtextfile
        Creates a mrecarray from data stored in the file `filename`.
    numpy.asarray
        Convert the input to an array.
    numpy.ndarray
        ndarray(shape, dtype=float, buffer=None, offset=0,
    numpy.recarray
        Construct an ndarray that allows field access using attributes.
    numpy.chararray
        chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
    numpy.pad
        Pads an array.
    numpy.asanyarray
        Convert the input to an ndarray, but pass ndarray subclasses through.
    numpy.copy
        Return an array copy of the given object.
    numpy.diag
        Extract a diagonal or construct a diagonal array.
    numpy.load
        Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
    numpy.sort
        Return a sorted copy of an array.
    numpy.array_equiv
        Returns True if input arrays are shape consistent and all elements equal.
    numpy.dtype
        Create a data type object.
    numpy.choose
        Construct an array from an index array and a set of arrays to choose from.
    numpy.nditer
        Efficient multi-dimensional iterator object to iterate over arrays.
    numpy.swapaxes
        Interchange two axes of an array.
    numpy.full_like
        Return a full array with the same shape and type as a given array.
    numpy.ones_like
        Return an array of ones with the same shape and type as a given array.
    numpy.empty_like
        Return a new array with the same shape and type as a given array.
    numpy.ma.mrecords.MaskedRecords.__new__
        Create a new masked array from scratch.
    numpy.nan_to_num
        Replace nan with zero and inf with finite numbers.
    numpy.zeros_like
        Return an array of zeros with the same shape and type as a given array.
    numpy.asarray_chkfinite
        Convert the input to an array, checking for NaNs or Infs.
    numpy.diag_indices
        Return the indices to access the main diagonal of an array.
    numpy.chararray.tolist
        a.tolist()
    numpy.ma.choose
        Use an index array to construct a new array from a set of choices.
    numpy.savez_compressed
        Save several arrays into a single file in compressed ``.npz`` format.
    numpy.matlib.rand
        Return a matrix of random values with given shape.
    numpy.ma.empty_like
        Return a new array with the same shape and type as a given array.
    numpy.ma.make_mask_none
        Return a boolean mask of the given shape, filled with False.
    numpy.ma.mrecords.fromrecords
        Creates a MaskedRecords from a list of records.
    numpy.around
        Evenly round to the given number of decimals.
    numpy.source
        Print or write to a file the source code for a NumPy object.
    numpy.diagonal
        Return specified diagonals.
    numpy.einsum_path
        Evaluates the lowest cost contraction order for an einsum expression by
    numpy.histogram2d
        Compute the bi-dimensional histogram of two data samples.
    numpy.fft.ifft
        Compute the one-dimensional inverse discrete Fourier Transform.
    numpy.fft.ifftn
        Compute the N-dimensional inverse discrete Fourier Transform.
    numpy.busdaycalendar
        A business day calendar object that efficiently stores information


```python
help(np.lookfor)
```

    Help on function lookfor in module numpy.lib.utils:
    
    lookfor(what, module=None, import_modules=True, regenerate=False, output=None)
        Do a keyword search on docstrings.
        
        A list of of objects that matched the search is displayed,
        sorted by relevance. All given keywords need to be found in the
        docstring for it to be returned as a result, but the order does
        not matter.
        
        Parameters
        ----------
        what : str
            String containing words to look for.
        module : str or list, optional
            Name of module(s) whose docstrings to go through.
        import_modules : bool, optional
            Whether to import sub-modules in packages. Default is True.
        regenerate : bool, optional
            Whether to re-generate the docstring cache. Default is False.
        output : file-like, optional
            File-like object to write the output to. If omitted, use a pager.
        
        See Also
        --------
        source, info
        
        Notes
        -----
        Relevance is determined only roughly, by checking if the keywords occur
        in the function name, at the start of a docstring, etc.
        
        Examples
        --------
        >>> np.lookfor('binary representation')
        Search results for 'binary representation'
        ------------------------------------------
        numpy.binary_repr
            Return the binary representation of the input number as a string.
        numpy.core.setup_common.long_double_representation
            Given a binary dump as given by GNU od -b, look for long double
        numpy.base_repr
            Return a string representation of a number in the given base system.
        ...
    


## Creating Arrays

### 1-Dimensional


```python
a = np.array([0,1,2,3])
a
```




    array([0, 1, 2, 3])




```python
a.ndim
```




    1




```python
a.shape
```




    (4,)




```python
len(a)
```




    4



### 2-D, 3-D and more 


```python
b = np.array([[0, 1, 2], [3, 4, 5]])
b
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
b.ndim
```




    2




```python
b.shape
```




    (2, 3)




```python
len(b)
```




    2




```python
c = np.array([[[1], [2]], [[3], [4]]])
c
```




    array([[[1],
            [2]],
    
           [[3],
            [4]]])




```python
c.shape
```




    (2, 2, 1)




```python
print("Number of dimensions in array c: ",c.ndim)
```

    Number of dimensions in array c:  3


### Evenly spaced 


```python
#Evenly spaced - notice how it always starts with 0 .. (n-1) and not 1!
a = np.arange(10)
a
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



### or,  number of points using linspace 


```python
# number of points
c = np.linspace(0, 1, 9) # start, end, number of points
c
```




    array([ 0.   ,  0.125,  0.25 ,  0.375,  0.5  ,  0.625,  0.75 ,  0.875,  1.   ])




```python
d = np.linspace(0, 1, 5, endpoint=False) # meaning it doesn't stop at 1.
d
```




    array([ 0. ,  0.2,  0.4,  0.6,  0.8])



### Common arrays 


```python
a = np.ones((3, 3)) # (3, 3) would be a tuple here
a
```




    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])




```python
type(a)
```




    numpy.ndarray




```python
b = np.zeros((2 ,2))
b
```




    array([[ 0.,  0.],
           [ 0.,  0.]])




```python
c = np.eye(3) # An identity matrix
c
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])




```python
d = np.diag(np.array([1,2,3,4]))
d
```




    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])



### Random numbers 


```python
e = np.random.rand(4) # uniform in [0, 1]
e
```




    array([ 0.42638175,  0.71417744,  0.3370914 ,  0.17486232])




```python
f = np.random.randn(4) #Gaussian
f
# type help(np.random.randn) to understand more
```




    array([-1.25793493,  0.97254164, -0.25065458,  0.0813759 ])




```python
np.random.seed(1234) #setting the random seed
help(np.random.seed)
```

    Help on built-in function seed:
    
    seed(...) method of mtrand.RandomState instance
        seed(seed=None)
        
        Seed the generator.
        
        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.
        
        Parameters
        ----------
        seed : int or array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.
        
        See Also
        --------
        RandomState
    



```python

```

## Exercise 1

**Create an array that looks like this:**

$$x = 
\begin{bmatrix}
    1 & 1 & 1 & 1 \\
    1 & 1 & 1 & 1 \\
    5 & 6 & 7 & 8 \\
    2 & 6 & 4 & 7 \\
\end{bmatrix}\tag{1}$$ and, 

**another one** that looks like this:

$$y = 
\begin{bmatrix}
    0. & 0. & 0. & 0. & 0.\\
    7. & 0. & 0. & 0. & 0.\\
    0. & 8. & 0. & 0. & 0.\\
    0. & 0. & 9. & 0. & 0.\\
    0. & 0. & 0. & 10. & 0.\\
    0. & 0. & 0. & 0. & 11.\\
\end{bmatrix}\tag{2}$$

and lastly,

**create** this simple array

$$\begin{bmatrix}
    0. & 0. & 0. & 0. & 0.\\
    0. & 0. & 0. & 0. & 0.\\
    1. & 0. & 0. & 0. & 0.\\
    0. & 1. & 0. & 0. & 0.\\
    0. & 0. & 1. & 0. & 0.\\
    0. & 0. & 0. & 1. & 0.\\
\end{bmatrix}\tag{3}$$


```python
help(np.eye)
help(np.diag)
```

    Help on function eye in module numpy.lib.twodim_base:
    
    eye(N, M=None, k=0, dtype=<class 'float'>)
        Return a 2-D array with ones on the diagonal and zeros elsewhere.
        
        Parameters
        ----------
        N : int
          Number of rows in the output.
        M : int, optional
          Number of columns in the output. If None, defaults to `N`.
        k : int, optional
          Index of the diagonal: 0 (the default) refers to the main diagonal,
          a positive value refers to an upper diagonal, and a negative value
          to a lower diagonal.
        dtype : data-type, optional
          Data-type of the returned array.
        
        Returns
        -------
        I : ndarray of shape (N,M)
          An array where all elements are equal to zero, except for the `k`-th
          diagonal, whose values are equal to one.
        
        See Also
        --------
        identity : (almost) equivalent function
        diag : diagonal 2-D array from a 1-D array specified by the user.
        
        Examples
        --------
        >>> np.eye(2, dtype=int)
        array([[1, 0],
               [0, 1]])
        >>> np.eye(3, k=1)
        array([[ 0.,  1.,  0.],
               [ 0.,  0.,  1.],
               [ 0.,  0.,  0.]])
    
    Help on function diag in module numpy.lib.twodim_base:
    
    diag(v, k=0)
        Extract a diagonal or construct a diagonal array.
        
        See the more detailed documentation for ``numpy.diagonal`` if you use this
        function to extract a diagonal and wish to write to the resulting array;
        whether it returns a copy or a view depends on what version of numpy you
        are using.
        
        Parameters
        ----------
        v : array_like
            If `v` is a 2-D array, return a copy of its `k`-th diagonal.
            If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
            diagonal.
        k : int, optional
            Diagonal in question. The default is 0. Use `k>0` for diagonals
            above the main diagonal, and `k<0` for diagonals below the main
            diagonal.
        
        Returns
        -------
        out : ndarray
            The extracted diagonal or constructed diagonal array.
        
        See Also
        --------
        diagonal : Return specified diagonals.
        diagflat : Create a 2-D array with the flattened input as a diagonal.
        trace : Sum along diagonals.
        triu : Upper triangle of an array.
        tril : Lower triangle of an array.
        
        Examples
        --------
        >>> x = np.arange(9).reshape((3,3))
        >>> x
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        
        >>> np.diag(x)
        array([0, 4, 8])
        >>> np.diag(x, k=1)
        array([1, 5])
        >>> np.diag(x, k=-1)
        array([3, 7])
        
        >>> np.diag(np.diag(x))
        array([[0, 0, 0],
               [0, 4, 0],
               [0, 0, 8]])
    


## Solutions 

## Exercise 1, Solutions 1, 2, 3

**Hint**: use help(np.diag) for info.

and also try out some more yourself!


```python
help(np.diag)
```

    Help on function diag in module numpy.lib.twodim_base:
    
    diag(v, k=0)
        Extract a diagonal or construct a diagonal array.
        
        See the more detailed documentation for ``numpy.diagonal`` if you use this
        function to extract a diagonal and wish to write to the resulting array;
        whether it returns a copy or a view depends on what version of numpy you
        are using.
        
        Parameters
        ----------
        v : array_like
            If `v` is a 2-D array, return a copy of its `k`-th diagonal.
            If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
            diagonal.
        k : int, optional
            Diagonal in question. The default is 0. Use `k>0` for diagonals
            above the main diagonal, and `k<0` for diagonals below the main
            diagonal.
        
        Returns
        -------
        out : ndarray
            The extracted diagonal or constructed diagonal array.
        
        See Also
        --------
        diagonal : Return specified diagonals.
        diagflat : Create a 2-D array with the flattened input as a diagonal.
        trace : Sum along diagonals.
        triu : Upper triangle of an array.
        tril : Lower triangle of an array.
        
        Examples
        --------
        >>> x = np.arange(9).reshape((3,3))
        >>> x
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        
        >>> np.diag(x)
        array([0, 4, 8])
        >>> np.diag(x, k=1)
        array([1, 5])
        >>> np.diag(x, k=-1)
        array([3, 7])
        
        >>> np.diag(np.diag(x))
        array([[0, 0, 0],
               [0, 4, 0],
               [0, 0, 8]])
    


### Exercise 1: Solution 1


```python
a = np.ones((4, 4), dtype=int)
a[3, 1] = 6
a[2, 3] = 2

print(a)
```

    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 2]
     [1 6 1 1]]


### Exercise 1: Solution 2


```python
b = np.zeros((6, 5))
b[1:] = np.diag(np.arange(7,12))
b
```




    array([[  0.,   0.,   0.,   0.,   0.],
           [  7.,   0.,   0.,   0.,   0.],
           [  0.,   8.,   0.,   0.,   0.],
           [  0.,   0.,   9.,   0.,   0.],
           [  0.,   0.,   0.,  10.,   0.],
           [  0.,   0.,   0.,   0.,  11.]])



### Exercise 1: Solution 3 


```python
y = np.eye(6, 5, k=-2, dtype=float)
y
```




    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.]])


