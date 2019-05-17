# Resnet50 Architecture

import keras.layers
import keras.regularizers

parameters = {
    "kernel_initializer": "he_normal"
}

class BatchNormalization(keras.layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        # return super.call, but set training
        return super(BatchNormalization, self).call(training=(not self.freeze), *args, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config


def basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
        >>> import keras_resnet.blocks
        >>> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)
        y = keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    Usage:
        >>> import keras_resnet.blocks
        >>> keras_resnet.blocks.bottleneck_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
        y = keras.layers.Activation("relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = keras.layers.Conv2D(filters * 4, (1, 1), use_bias=False, name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)
        y = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = keras.layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)
            shortcut = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
        y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y


def ResNet(inputs, blocks, block, include_top=True, classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.blocks
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> blocks = [2, 2, 2, 2]
        >>> block = keras_resnet.blocks.basic_2d
        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = keras.layers.ZeroPadding2D(padding=3, name="padding_conv1")(inputs)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
    x = keras.layers.Activation("relu", name="conv1_relu")(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    features = 64

    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = basic_2d(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0

        x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

        return keras.models.Model(inputs=inputs, outputs=x, *args, **kwargs)
    else:
        # Else output each stages features
        return keras.models.Model(inputs=inputs, outputs=outputs, *args, **kwargs)


def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = keras_resnet.models.ResNet50(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d(64), include_top=include_top, classes=classes, *args, **kwargs)


# ResNet Backbone
def resnet_retinanet(num_classes, backbone='resnet50', inputs= None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)
