GPU Backend Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backends
-------------

.. autoclass:: gpubackendtools.gpubackendtools.Backend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.CpuBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda11xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda12xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda13xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendMethods
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendsManager
    :members:
    :show-inheritance:

Backend Functions
--------------------

.. autofunction:: gpubackendtools.get_backend
.. autofunction:: gpubackendtools.get_first_backend
.. autofunction:: gpubackendtools.has_backend

Globals & Configuration
------------------------

.. autoclass:: gpubackendtools.globals.Globals
    :members:
    :show-inheritance:

Interpolation
--------------

.. autoclass:: gpubackendtools.interpolate.CubicSplineInterpolant
    :members:
    :show-inheritance:
    :inherited-members:

Exceptions
-------------

.. autoclass:: gpubackendtools.exceptions.GPUBACKENDTOOLSException
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.BackendUnavailableException
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.CudaException
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.CuPyException
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.MissingDependency
    :members:
    :show-inheritance:
