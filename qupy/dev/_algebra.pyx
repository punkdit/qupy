
from cpython.object cimport Py_EQ, Py_NE

DEF ONE = 1.0
DEF ZERO = 0.0
DEF EPSILON = 1e-8


cdef class Tensor

from qupy.tool import fstr


cdef class Algebra:
    cdef public int dim
    cdef public object names
    cdef public object coefs 
    cdef public object basis
    cdef public object lookup

    def __init__(self, int dim, names, coefs):
        self.dim = dim
        self.names = names
        self.coefs = coefs

        basis = []
        for i in range(dim):
            op = Tensor(self)
            op[(i,)] = ONE
            basis.append(op)
        self.basis = basis

        lookup = {}
        for i in range(dim):
          for j in range(dim):
            op = Tensor(self)
            for k in range(dim):
                val = coefs.get((i, j, k), ZERO)
                if abs(val)>EPSILON:
                    op[(k,)] = val
            lookup[i, j] = op
        self.lookup = lookup

    def parse(self, desc):
        n = len(desc)
        idxs = tuple(self.names.index(c) for c in desc)
        op = Tensor(self)
        op[idxs] = ONE
        return op

    def __getattr__(self, attr):
        if attr in self.names:
            return self.parse(attr)
        raise AttributeError

    def get_zero(self, grade):
        op = Tensor(self)
        return op


cdef class Tensor:
    "Tree shaped data-structure"

    cdef public Algebra algebra
    cdef public object children
    cdef public object value
    cdef object _keys, _values, _items

    def __init__(Tensor self, Algebra algebra):
        self.algebra = algebra
        self.children = [None]*algebra.dim
        self.value = 0.0
        self._keys = None
        self._values = None
        self._items = None

    def __str__(Tensor self):
        ss = []
        algebra = self.algebra
        keys = self.get_keys()
        keys.sort()
        for k in keys:
            v = self[k]
            s = ''.join(algebra.names[ki] for ki in k)
            if abs(v) < EPSILON:
                continue
            elif abs(v-1) < EPSILON:
                pass
            elif abs(v+1) < EPSILON:
                s = "-"+s
            else:
                s = fstr(v)+"*"+s
            ss.append(s)
        ss = '+'.join(ss) or "0"
        ss = ss.replace("+-", "-")
        return ss

    def __repr__(Tensor self):
        return self.__str__()

    def get_zero(Tensor self):
        return Tensor(self.algebra)

    def __getitem__(Tensor self, key):
        cdef int i
        cdef Tensor child, _child
        child = self
        for i in key:
            _child = child.children[i]
            if _child is None:
                return ZERO
            child = _child
        return child.value

    @property
    def grade(Tensor self):
        cdef int i
        cdef Tensor child, _child
        for i in range(self.algebra.dim):
            child = self.children[i]
            if child is not None:
                return child.grade+1
        return 0

    def iadditem(Tensor self, key, object value):
        # increment by value at key
        cdef int i
        cdef Tensor child, _child
        child = self
        for i in key:
            _child = child.children[i]
            if _child is None:
                _child = Tensor(self.algebra)
                child.children[i] = _child
            child = _child
        child.value += value

        # flush cache
        self._keys = None
        self._values = None
        self._items = None

    def __setitem__(Tensor self, key, object value):
        # set value at key
        cdef int i
        cdef Tensor child, _child
        child = self
        for i in key:
            _child = child.children[i]
            if _child is None:
                _child = Tensor(self.algebra)
                child.children[i] = _child
            child = _child
        child.value = value

        # flush cache
        self._keys = None
        self._values = None
        self._items = None

    cdef object get_keys(Tensor self):
        cdef Tensor child
        cdef int i
        if self._keys is not None:
            return self._keys
        keys = []
        if self.value != ZERO:
            keys.append(())
        for i from 0<=i<self.algebra.dim:
            child = self.children[i]
            if child is None:
                continue
            for key in child.get_keys():
                keys.append((i,)+key)
        self._keys = keys
        return keys

    cdef object get_values(Tensor self):
        cdef Tensor child
        cdef int i
        if self._values is not None:
            return self._values
        values = []
        if self.value != ZERO:
            values.append(self.value)
        for i from 0<=i<self.algebra.dim:
            child = self.children[i]
            if child is None:
                continue
            for value in child.get_values():
                values.append(value)
        self._values = values
        return values

    cdef object get_items(Tensor self):
        cdef Tensor child
        cdef int i
        if self._items is not None:
            return self._items
        items = []
        if self.value != ZERO:
            items.append(((), self.value))
        for i from 0<=i<self.algebra.dim:
            child = self.children[i]
            if child is None:
                continue
            for (key, value) in child.get_items():
                items.append(((i,)+key, value))
        self._items = items
        return items

    def copy(Tensor self):
        op = Tensor(self.algebra)
        for (k, v) in self.get_items():
            op[k] = v
        return op

    def keys(Tensor self):
        return self.get_keys()

    def values(Tensor self):
        return self.get_values()

    def items(Tensor self):
        return self.get_items()

    def norm(self):
        return sum(abs(val) for val in self.get_values())

    def eq(self, other):
        if not isinstance(other, Tensor) and other==0:
            other = self.get_zero()
        return (self-other).norm() < EPSILON

    def ne(self, other):
        if not isinstance(other, Tensor) and other==0:
            other = self.get_zero()
        return (self-other).norm() > EPSILON

    def __richcmp__(self, other, int idx):
        if idx != Py_EQ and idx != Py_NE:
            assert 0, idx
    
        if other is None:
            return idx == Py_NE
    
        if not isinstance(other, Tensor) and other==0:
            other = self.get_zero()
        r = (self-other).norm()
        if idx==Py_EQ:
            return r<EPSILON
        elif idx==Py_NE:
            return r>EPSILON

    def __add__(Tensor self, Tensor other):
        #assert self.grade == other.grade # i guess this is not necessary...
        cdef Tensor op = self.copy()
        for (k, v) in other.get_items():
            op.iadditem(k, v)
        return op

    def __sub__(Tensor self, Tensor other):
        #assert self.grade == other.grade
        cdef Tensor op = self.copy()
        for (k, v) in other.get_items():
            op.iadditem(k, -v)
        return op

    def rmul(Tensor self, r):
        cdef Tensor op = Tensor(self.algebra)
        for (k, v) in self.get_items():
            op[k] = complex(r)*v
        return op

    def __neg__(Tensor self):
        cdef Tensor op = Tensor(self.algebra)
        for (k, v) in self.get_items():
            op[k] = -v
        return op

    def __matmul__(Tensor self, Tensor other):
        assert self.algebra is other.algebra
        cdef Tensor op = Tensor(self.algebra)
        items = self.get_items()
        jtems = other.get_items()
        for (k1, v1) in items:
          for (k2, v2) in jtems:
            k = k1+k2
            assert op[k] == ZERO
            op[k] = v1*v2
        return op

    def __mul__(_self, _other):
        if not isinstance(_self, Tensor):
            return _other.rmul(_self)
        cdef Tensor self = _self
        cdef Tensor other = _other
        assert self.algebra is other.algebra
        cdef Tensor op1
        cdef Tensor op = Tensor(self.algebra)

        items = self.get_items()
        jtems = other.get_items()

        n = self.grade

        for (idx, val) in items:
          if abs(val)<EPSILON:
              continue
          for (jdx, wal) in jtems:
            if abs(wal)<EPSILON:
                continue
            r = val*wal
            #key = ()
            key = []
            for i in range(n):
                op1 = self.algebra.lookup[idx[i], jdx[i]]
                k1, v1 = op1.get_items()[0]
                #key = key + k1
                key.append(k1[0])
                r *= v1
            op.iadditem(key, r)

        return op

    def subs(self, rename):
        the_op = Tensor(self.algebra) # zero
        algebra = self.algebra
        for (k, v) in self.get_items():
            final = None
            for ki in k:
                c = algebra.names[ki]
                op = rename.get(c)
                if op is None:
                    op = Tensor(self.algebra)
                    op[(ki,)] = ONE
                if final is None:
                    final = op
                else:
                    final = final @ op # tensor
            the_op = the_op + complex(v)*final # ARRGGGHHH !!
        return the_op


        

def build_algebra(names, rel):
    names = list(names)
    assert names[0] == "I" # identity
    dim = len(names)
    coefs = {} # structure coefs
    coefs[0, 0, 0] = ONE
    for i in range(1, dim):
        coefs[0, i, i] = ONE
        coefs[i, 0, i] = ONE

    eqs = rel.split()
    for eq in eqs:
        lhs, rhs = eq.split("=")
        A, B = lhs.split("*")
        i = names.index(A)
        j = names.index(B)
        rhs, C = rhs[:-1], rhs[-1]
        k = names.index(C)
        val = None
        if not rhs:
            val = ONE
        elif rhs == "-":
            val = -ONE
        else:
            assert 0, repr(eq)
        assert coefs.get((i, j, k)) is None
        coefs[i, j, k] = val 

    #A = numpy.zeros((dim, dim, dim), dtype=scalar.dtype)
    #for key, value in coefs.items():
    #    A[key] = value
    algebra = Algebra(dim, names, coefs)
    return algebra





