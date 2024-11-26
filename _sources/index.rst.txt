
aws.osml.model_runner
=====================

This package contains an application used to orchestrate the execution of ML models on large satellite images. The
application monitors an input queue for processing requests, decomposes the image into a set of smaller regions and
tiles, invokes an ML model endpoint with each tile, and finally aggregates all the results into a single output. The
application itself has been containerized and is designed to run on a distributed cluster of machines collaborating
across instances to process images as quickly as possible.

.. toctree::
   :maxdepth: 4


Indices and tables
__________________

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
