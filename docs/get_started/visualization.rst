*************
Visualization
*************


How to visualize the representations using t-SNE?
===================================================================


How to visualize the segmentation predictions?
===================================================================
For each segmentation algorithms, we've implemented the visualization code. All you need to do is set `--debug` during training.
For instance, in the directory `examples-da/segmentation`,

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python source_only.py data/GTA5 data/Cityscapes \
        -s GTA5 -t Cityscapes --log logs/src_only/gtav2cityscapes --debug

Then you can find visualization images under `logs/src_only/gtav2cityscapes/visualize/`.

.. figure:: ../_static/images/visualization/segmentation_image.png
    :width: 300

    Cityscapes image.

.. figure:: ../_static/images/visualization/segmentation_pred.png
    :width: 300

    Segmentation predictions.

.. figure:: ../_static/images/visualization/segmentation_label.png
    :width: 300

    Segmentation labels.


Translation model such as CycleGAN will save images by default. Here is the translation results from source style to target style.


.. figure:: ../_static/images/visualization/cyclegan_real_S.png
    :width: 300

    Source images.

.. figure:: ../_static/images/visualization/cyclegan_fake_T.png
    :width: 300

    Source image in target style.



How to visualize the keypoint detection predictions?
===================================================================
For each keypoint detection algorithms, we've implemented the visualization code. All you need to do is set `--debug` during training.
For instance, in the directory `examples-da/keypoint_detection`,

.. code-block:: shell

    CUDA_VISIBLE_DEVICES=0 python source_only.py data/RHD data/H3D_crop \
        -s RenderedHandPose -t Hand3DStudio --log logs/baseline/rhd2h3d --debug --seed 0

Then you can find visualization images under `logs/baseline/rhd2h3d/visualize/`.

.. figure:: ../_static/images/visualization/keypoint_detection.jpg
    :width: 300
