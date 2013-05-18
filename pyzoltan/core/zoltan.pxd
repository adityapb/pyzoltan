cimport mpi4py.MPI as mpi
from mpi4py cimport mpi_c as mpic

# Zoltan imports
from pyzoltan.czoltan cimport czoltan
from pyzoltan.czoltan.czoltan cimport Zoltan_Struct

# Zoltan type imports
from pyzoltan.czoltan.czoltan_types cimport ZOLTAN_ID_PTR, ZOLTAN_ID_TYPE, \
     ZOLTAN_OK, ZOLTAN_WARN, ZOLTAN_FATAL, ZOLTAN_MEMERR

# NUMPY
import numpy as np
cimport numpy as np

# Carrays
from carray cimport UIntArray, IntArray, LongArray, DoubleArray

# Error checking for Zoltan
cdef _check_error(int ierr)

# Pointer to the Zoltan struct
cdef struct _Zoltan_Struct:
    czoltan.Zoltan_Struct* zz

cdef class PyZoltan:
    # dimension
    cdef public int dim

    # version number
    cdef public double version
    
    # mpi.Comm object and associated rank and size
    cdef public object comm
    cdef public int rank, size
    
    # Pointer to the Zoltan structure upon creation
    cdef _Zoltan_Struct _zstruct

    # string to store the current load balancing method
    cdef public str lb_method

    # Arrays returned by Zoltan
    cdef public UIntArray exportGlobalids
    cdef public UIntArray exportLocalids
    cdef public IntArray exportProcs

    cdef public UIntArray importGlobalids
    cdef public UIntArray importLocalids
    cdef public IntArray importProcs

    # the number of objects to import/export
    cdef public int numImport, numExport

    cdef public np.ndarray procs             # processors of range size
    cdef public np.ndarray parts             # partitions of range size

    # General Zoltan parameters (refer the user guide)
    cdef public str ZOLTAN_DEBUG_LEVEL
    cdef public str ZOLTAN_OBJ_WEIGHT_DIM
    cdef public str ZOLTAN_EDGE_WEIGHT_DIM
    cdef public str ZOLTAN_RETURN_LISTS

    ###############################################################
    # Member functions
    ###############################################################
    # after a load balance, copy the Zoltan allocated lists to local
    # numpy arrays. The Zoltan lists are freed after a call to LB_Balance
    cdef _set_Zoltan_lists(
        self,                                           
        int numExport,                          # number of objects to export
        ZOLTAN_ID_PTR _exportGlobal,            # global indices of export objects
        ZOLTAN_ID_PTR _exportLocal,             # local indices of export objects
        int* _exportProcs,                      # target processors to export
        int numImport,                          # number of objects to import
        ZOLTAN_ID_PTR _importGlobal,            # global indices of import objects
        ZOLTAN_ID_PTR _importLocal,             # local indices of import objects
        int* _importProcs                       # target processors to import
        )

    # Invert the lists after computing remote particles
    cpdef Zoltan_Invert_Lists(self)    

# User defined data for the RCB, RIB and HSFC methods
cdef struct CoordinateData:
    int dim
    int numGlobalPoints
    int numMyPoints

    ZOLTAN_ID_PTR myGlobalIDs
    double* x
    double* y
    double *z
    
cdef class ZoltanGeometricPartitioner(PyZoltan):
    # data arrays for the coordinates
    cdef public DoubleArray x, y, z

    # data array for the global indices
    cdef public UIntArray gid

    # User defined structure to hold the coordinate data for the Zoltan interface
    cdef CoordinateData _cdata

    # number of global and local objects
    cdef public int num_global_objects, num_local_objects

    # ZOLTAN parameters for Geometric partitioners
    cdef public str ZOLTAN_KEEP_CUTS
    cdef public str ZOLTAN_LB_METHOD
    
