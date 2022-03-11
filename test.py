import paddle
import paddle.nn as nn
import miziha
import numpy as np


def droppath_test(*shape):
    n, c, h, w = shape
    tmp = paddle.to_tensor(np.random.rand(n, c, h, w), dtype='float32')
    dp = miziha.DropPath(0.5)
    out = dp(tmp)
    print(out)

def window_partition_test(*shape):
    n, c, h, w = shape
    x = paddle.to_tensor(np.random.rand(n, h, w, c), dtype='float32')
    window_size = 7
    out = miziha.windows_partition(x, window_size)
    print(out.shape)

def SwinT_test():
    # 创建测试数据
    test_data = paddle.ones([2, 32, 224, 224]) #[N, C, H, W]
    print(f'输入尺寸:{test_data.shape}')

    # 创建SwinT层
    '''
    参数：
    in_channels: 输入通道数，同卷积
    out_channels: 输出通道数，同卷积

    以下为SwinT独有的，类似于卷积中的核大小，步幅，填充等
    input_resolution: 输入图像的尺寸大小
    num_heads: 多头注意力的头数，应该设置为能被输入通道数整除的值
    window_size: 做注意力运算的窗口的大小，窗口越大，运算就会越慢
    qkv_bias: qkv的偏置，默认None
    qk_scale: qkv的尺度，注意力大小的一个归一化，默认None      #Swin-V1版本
    dropout: 默认None 
    attention_dropout: 默认None 
    droppath: 默认None 
    downsample: 下采样，默认False，设置为True时，输出的图片大小会变为输入的一半
    '''
    swint1 = miziha.SwinT(in_channels=32, out_channels=128, input_resolution=(224, 224), num_heads=8, window_size=7,
                          downsample=False)
    swint2 = miziha.SwinT(in_channels=32, out_channels=128, input_resolution=(224, 224), num_heads=8, window_size=7,
                          downsample=True)
    conv1 = nn.Conv2D(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)

    # 前向传播，打印输出形状
    output1 = swint1(test_data)
    output2 = swint2(test_data)
    output3 = conv1(test_data)

    print(f'SwinT的输出尺寸:{output1.shape}')
    print(f'下采样的SwinT的输出尺寸:{output2.shape}')  # 下采样
    print(f'Conv2D的输出尺寸:{output3.shape}')


# droppath_test(3, 4, 5, 5)

# window_partition_test(2, 3, 224, 224)

SwinT_test()