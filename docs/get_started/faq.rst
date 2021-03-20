*************
FAQ
*************

1. How to build the doc?
=========================

Sometimes, the online doc is not consistent with the latest code. In this case, you can build the doc by yourself.
First, you need to install sphinx, which is used to build doc.

.. code-block:: shell

    pip install -U sphinx

Second, you need to download a pytorch-style `html.zip <https://cloud.tsinghua.edu.cn/f/4d6b594de2694b399fb9/?dl=1>`_
into directory ``docs/build`` and unzip it. Some browsers may identify it as malicious file. You can safely ignore those warnings.

Then, in the directory ``docs`` run the following command

.. code-block:: shell

    make html

Also, warnings during ``make`` process doesn't matter and can be ignored.

Finally, you can open the docs in ``docs/build/html/index.html``
