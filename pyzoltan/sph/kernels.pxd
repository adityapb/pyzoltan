from point cimport cPoint

##############################################################################
#`Kernel`
##############################################################################
cdef class Kernel:
    cdef readonly int dim
    cdef readonly double radius
    cdef readonly double fac

    cdef double function(self, cPoint xi, cPoint xj, double h)
    cdef gradient(self, cPoint xi, cPoint xj, double h, cPoint grad)
    cdef double gradient_h(self, cPoint xi, cPoint xj, double h)

    cdef double _function(self, double q)
    cdef double _gradient(self, double q)

cdef class CubicSpline(Kernel):
    pass

cdef class Gaussian(Kernel):
    pass
