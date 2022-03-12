import tensorflow as tf
from conformer_tf import ConformerConvModule

layer = ConformerConvModule(
    dim=512,
    causal=False,  # whether it is auto-regressive
    expansion_factor=2,  # what multiple of the dimension to expand for the depthwise convolution
    kernel_size=31,
    dropout=0.
)

x = tf.random.normal([1, 1024, 512])
x = layer(x) + x  # (1, 1024, 512)
