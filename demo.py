import torch.cuda
from torchinfo import summary

from project_utils import *


def other_simple():
    # torch.Tensor：使用它创建的张量对象没有指定数据类型，因此其默认为浮点数类型（float32），其值取决于内存中的随机数据。
    # torch.tensor：根据给定的数据创建一个张量对象，并自动推断数据类型。可以接受多种数据类型作为输入参数，例如列表、元组、数组等。
    # tensor([1.0561e-38, 1.0194e-38, 9.2755e-39, 8.9082e-39, 8.4490e-39, 9.2755e-39]) torch.Size([6])
    # 传入一个整数 n 时，torch.Tensor 认识 n 是一维张量的元素个数，并随机初始化
    print(torch.Tensor(6), torch.Tensor(6).shape)
    # tensor(6) torch.Size([])
    # torch.Size([]) 和 torch.Size([0]) 的区别，前者是一个标量，不能迭代，后者是一个数组，只不过没有元素，形如 []，可以迭代
    # torch.tensor则会将n视作一个数字而不是元素个数。例如：
    print(torch.tensor(6), torch.tensor(6).shape)
    # tensor([1., 2., 3.])
    print(torch.Tensor([1, 2, 3]))
    # tensor([1, 2, 3])
    print(torch.tensor([1, 2, 3]))

    # torch.arange(start=0, end, step=1)
    # 同 python 的 range，前开后闭
    assert_tensor_equal(torch.arange(1, 10, 3), [1, 4, 7])
    assert_tensor_equal(torch.arange(1, 1, 3), [])

    # reshape 会视情况返回 view 或者是新的 copy，不能依赖这一点
    # 但 view 可能会失败，因为必须是 strided
    assert_tensor_shape_equal(torch.arange(1, 10).reshape((3, 3)), (3, 3))

    a = get_a_sample_tensor((2, 3))
    # transpose 是将两个维度进行转置，如果 strided 则原位，否则返回 copy
    b = a.transpose(0, 1)
    # view 相当于是 reshape，注意 view 可能会执行失败，如果 tensor 不是 strided，而 reshape 则不会失败，如果不是 strided 就返回 copy
    c = a.view(3, 2)
    assert_tensor_equal(b, [[0, 3], [1, 4], [2, 5]])
    assert_tensor_equal(c, [[0, 1], [2, 3], [4, 5]])
    assert_tensor_shape_equal(b, c)
    assert not torch.equal(b, c)

    # linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long
    # 如果不加 dim，就不能 keepdim=True，因为这个时候已经是一个标量，没法扩展维度，已经恢复不了了
    a = get_a_sample_tensor((2, 3))
    # 如果 norm 不传入任何参数，则等价于将 tensor 展平，返回所有元素的 norm
    # norm(p: Optional[Union[float, str]] = "fro", dim=None, keepdim=False, dtype=None,
    assert_tensor_close(a.norm(), 7.4162)
    assert_tensor_close(a.norm(keepdim=True), [[7.4162]])
    assert_tensor_close(a.norm(dim=0), [3.0000, 4.1231, 5.3852])
    assert_tensor_close(a.norm(dim=(0, 1)), 7.4162)

    # tf.expand: 将张量广播到新的形状。
    # 注意： 只能对维度值为 1 的维度进行扩展，无需扩展的维度，维度值不变，对应位置可写上原始维度大小或直接写作 -1
    # 且扩展的Tensor不会分配新的内存，只是原来的基础上创建新的视图并返回，返回的张量内存是不连续的。
    # 类似于numpy中的broadcast_to函数的作用。如果希望张量内存连续，可以调用contiguous函数。
    # expand函数可能导致原始张量的升维，其作用在张量前面的维度上(在tensor的低维增加更多维度)，因此通过expand函数可将张量数据复制多份（可理解为沿着第一个batch的维度上）
    # expand_as 可视为 expand 的另一种表达，其size通过函数传递的目标张量的size来定义。
    a = torch.arange(6).reshape((1, 1, 2, 3))
    assert_tensor_shape_equal(a.expand(2, -1, -1, -1), [2, 1, 2, 3])
    assert_tensor_shape_equal(a.squeeze(), [2, 3])
    assert_tensor_shape_equal(a.squeeze(0), [1, 2, 3])
    assert_tensor_shape_equal(a.squeeze((0, 1)), [2, 3])
    # 如果 squeeze 某一位不是 1，会兼容不处理，可以传入 tuple
    assert_tensor_shape_equal(a.squeeze((0, 1, 2)), [2, 3])
    # unsqueeze 不能传入 tuple
    assert_tensor_shape_equal(a.unsqueeze(2), [1, 1, 1, 2, 3])

    # tensor.repeat()：和expand()作用类似，均是将tensor广播到新的形状。
    # 注意：不允许使用维度 -1，1 即为不变。
    # 前文提及expand仅能作用于单数维，那对于非单数维的拓展，那就需要借助于repeat函数了。
    # tensor.repeat(*sizes)
    # 参数*sizes指定了原始张量在各维度上复制的次数。整个原始张量作为一个整体进行复制，这与Numpy中的repeat函数截然不同，而更接近于tile函数的效果。
    # 与expand不同，repeat函数会真正的复制数据并存放于内存中。repeat开辟了新的内存空间，torch.repeat返回的张量在内存中是连续的
    assert_tensor_shape_equal(a.repeat(1, 2, 3, 4), [1, 2, 6, 12])


# 各种乘法
def check_mul():
    a = get_a_sample_tensor((2, 3))
    b = get_a_sample_tensor((2, 3))
    c = get_a_sample_tensor((3, 2))
    d = get_a_sample_tensor((1,))
    f = get_a_sample_tensor((3,))
    g = get_a_sample_tensor((3,))
    h = get_a_sample_tensor((2, 2, 3))
    x = get_a_sample_tensor((10, 1, 3, 4))
    y = get_a_sample_tensor((10, 3, 4, 5))
    z = get_a_sample_tensor((10, 3, 4, 5))

    # torch.mul 逐个元素相乘，可以进行广播，简写是 *
    # 广播机制：
    # 1.如果维度个数不同，则在维度较少的左边补1，使得维度的个数相同。
    # 2.各维度的维度大小不同时，如果有维度为1的，直接将该维拉伸至维度相同
    assert_tensor_shape_equal(torch.mul(a, b), (2, 3))
    assert_tensor_shape_equal(torch.mul(a, d), (2, 3))
    # The size of tensor a (3) must match the size of tensor b (6) at non-singleton dimension 1
    # 广播只能广播有 1 的，不能智能的求公倍数
    # assert_tensor_shape_equal(torch.mul(a, e), (2, 3))

    # torch.matmul 矩阵相乘，简写是 @
    # vector * vector，得到一个标量
    # 1D * 1D = 0D
    assert_tensor_shape_equal(torch.matmul(f, g), tuple())
    # matrix * vector，得到一个 vector
    # 2D * 1D = 1D
    assert_tensor_shape_equal(torch.matmul(a, g), (2,))
    # 3D * vector = 2D
    assert_tensor_shape_equal(torch.matmul(h, g), (2, 2))
    # (..., a, b)D * (..., b, c)D = (..., a, c)D，最后2D 必须满足矩阵乘法的条件，前面必须一致，这个过程中可以触发广播机制
    assert_tensor_shape_equal(torch.matmul(x, y), (10, 3, 3, 5))
    assert_tensor_shape_equal(torch.matmul(x, z), (10, 3, 3, 5))

    # torch.mm，矩阵相乘，不会进行广播，必须满足矩阵相乘维数条件,两矩阵最多是2维
    assert_tensor_shape_equal(torch.mm(a, c), (2, 2))

    # torch.bmm，批矩阵相乘，不会进行广播，必须满足矩阵相乘维数条件，a,b最多只能3维，且a,b中必须包含相同的矩阵个数即a,b第一维度必须相同
    assert_tensor_shape_equal(torch.bmm(torch.unsqueeze(a, 0), torch.unsqueeze(c, 0)), (1, 2, 2))

    # torch.dot(a,b)，向量点积，两向量相乘相加得到一个标量，必须都是一维的
    assert_tensor_shape_equal(torch.dot(f, g), tuple())


# dim (int or tuple of python:ints) – the dimension or dimensions to reduce.
# 在哪些维度上面做平均
# keepdim (bool) – whether the output tensor has dim retained or not.
# 是否保持和原来一样的维度，默认是 False，注意维度表示坐标系，即三维表示有三个方向，shape 表示每个方向多大
def check_mean_op():
    # 2 * 3 * 4
    array = np.resize(np.array(range(1, 25)), (2, 3, 4))
    tensor = torch.tensor(array, dtype=torch.float)

    # 注意 tensor.mean() 是标量，不能用 to_tuple(0) 来比较
    assert_tensor_shape_equal(tensor.mean(), tuple())
    # 如果不加 dim，就不能 keepdim=True，因为这个时候已经是一个标量，没法扩展维度，已经恢复不了了
    # tensor.mean(keepdim=True)
    assert_tensor_shape_equal(tensor.mean(dim=(1,)), (2, 4))
    assert_tensor_shape_equal(tensor.mean(dim=(1,), keepdim=True), (2, 1, 4))
    assert_tensor_shape_equal(tensor.mean(dim=(0,)), (3, 4))
    assert_tensor_shape_equal(tensor.mean(dim=(0,), keepdim=True), (1, 3, 4))
    assert_tensor_shape_equal(tensor.mean(dim=(0, 1)), 4)
    assert_tensor_shape_equal(tensor.mean(dim=(0, 1), keepdim=True), (1, 1, 4))


# unbiased (bool) – whether to use Bessel’s correction
# 是否开启 贝塞尔校正，默认是 True，形如 np.std() 中的 ddof=1，Means Delta Degrees of Freedom
# 在统计学中，贝塞尔校正是在样本的方差和标准差的公式中用n-1来代替n。这个方法校正了样本方差/样本标准差，与总体方差/样本标准差之间的误差。
# 举一个例子，如果一个数据集满足高斯分布（Normal Distribution），那当我们提取样本的时候，数据基本上会集中在中间的部分，而边缘值的数目可能会比较少，
# 所以最后得到的样本方差和样本标准差会比总体要小。为了修正这个偏差，在计算样本的方差和标准差时，我们将使用 n-1 代替 n。这样处理后最直接的结果是，公式中的分母变小，得到的结果将会变大，能够更加准确地通过该样本预测总体的情况。
def check_std_op():
    # 2 * 3 * 4
    array = np.array(range(1, 4))
    # sqrt((1 + 0 + 1) / 3) = 0.816496580927726
    assert_close(np.std(array), 0.8165)
    # sqrt((1 + 0 + 1) / (3 - 1)) = 1.0
    assert_equal(np.std(array, ddof=1), 1)

    tensor = torch.tensor(array, dtype=torch.float)
    assert_tensor_equal(tensor.std(), 1)
    assert_tensor_close(tensor.std(unbiased=False), 0.8165)


# per channel
# 每一个 C =  channel 内，对整个 mini-batch 的归一化，收缩几维，就有多少对 (γ, β)
# 除了 C 都收缩
# normalized_shape = C, 每一个 normalized_shape 有一对 (γ, β)
# (N, C) -> (1, C)
# nlp, C 是 embedding 的维度，l 是样本长度
# (N, C, L) -> (1, C, 1)
# cv，C 是通道数，H、W 是长宽
# (N, C, H, W) -> (1, C, 1, 1)
def check_batch_norm():
    # Applies Batch Normalization over a 2D or 3D input
    # 一维 BN，针对 2d 或者 3d input
    # num_features – number of features or channels C of the input
    # eps – a value added to the denominator for numerical stability. Default: 1e-5
    # momentum – the value used for the running_mean and running_var computation.
    # Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    # affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True
    # track_running_stats – a boolean value that when set to True, this module tracks the running mean and variance,
    # and when set to False, this module does not track such statistics, and initializes statistics buffers
    # running_mean and running_var as None. When these buffers are None, this module always uses batch statistics.
    # in both training and eval modes. Default: True
    c_index = 1

    # (N, C) -> (1, C)
    shape = (2, 3)
    c = shape[c_index]
    other_index_list = get_index_exclude_index(shape, c_index)
    x = get_a_sample_tensor(shape)
    x_bn = nn.BatchNorm1d(c)(x)
    x_bn_check = get_tensor_norm(x, other_index_list)
    assert_tensor_close(x_bn, x_bn_check)

    # (N, C, L) -> (1, C, 1)
    shape = (2, 3, 4)
    c = shape[c_index]
    x = get_a_sample_tensor(shape)
    x_bn = nn.BatchNorm1d(c)(x)
    other_index_list = get_index_exclude_index(shape, c_index)
    # 对比 nn.BatchNorm1d(c)，对除了 c 以外的维度全部收缩
    x_bn_check = get_tensor_norm(x, other_index_list)
    assert_tensor_close(x_bn, x_bn_check)

    # Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    # 二维 BN，针对 4d input
    # (N, C, H, W) -> (1, C, 1, 1)
    shape = (2, 3, 4, 5)
    c = shape[c_index]
    other_index_list = get_index_exclude_index(shape, c_index)
    x = get_a_sample_tensor(shape)
    x_bn = nn.BatchNorm2d(c)(x)
    # 对比 nn.BatchNorm1d(c)，对除了 c 以外的维度全部收缩
    x_bn_check = get_tensor_norm(x, other_index_list)
    assert_tensor_close(x_bn, x_bn_check)


# per sample per layer
# 只对指定的后几维做收缩
# normalized_shape = 最后几个维度，收缩几维，就有多少对 (γ, β)
# (N, C) -> (N, 1)
# nlp
# (N, L, C) -> (N, L, 1)
# cv
# (N, C, H, W) -> (N, 1, 1, 1)

# NLP Example
# embedding = torch.randn(batch, sentence_length, embedding_dim)
# layer_norm = nn.LayerNorm(embedding_dim)
# layer_norm(embedding)

# Image Example
# N, C, H, W = 20, 5, 10, 10
# input = torch.randn(N, C, H, W)
# layer_norm = nn.LayerNorm([C, H, W])
def check_layer_norm():
    # Applies Layer Normalization over a mini-batch of inputs
    # 任意维 LN
    # 和 BN 不同，LN 中的 mean 和 std 是由当前样本决定的，不更新

    # (N, C) -> (N, 1)
    shape = (2, 3)
    last_dimensions = [-1]
    normalized_shape = to_tuple(get_list_from_index(shape, last_dimensions))
    x = get_a_sample_tensor(shape)
    x_bn = nn.LayerNorm(normalized_shape)(x)
    x_bn_check = get_tensor_norm(x, last_dimensions)
    assert_tensor_close(x_bn, x_bn_check)

    # (N, L, C) -> (N, L, 1)
    shape = (2, 3, 4)
    last_dimensions = [-1]
    normalized_shape = to_tuple(get_list_from_index(shape, last_dimensions))
    x = get_a_sample_tensor(shape)
    x_bn = nn.LayerNorm(normalized_shape)(x)
    x_bn_check = get_tensor_norm(x, last_dimensions)
    assert_tensor_close(x_bn, x_bn_check)

    # (N, L, C) -> (N, 1, 1)
    shape = (2, 3, 4)
    last_dimensions = [-2, -1]
    normalized_shape = to_tuple(get_list_from_index(shape, last_dimensions))
    x = get_a_sample_tensor(shape)
    x_bn = nn.LayerNorm(normalized_shape)(x)
    x_bn_check = get_tensor_norm(x, last_dimensions)
    assert_tensor_close(x_bn, x_bn_check)

    # (N, C, H, W) -> (N, 1, 1, 1)
    shape = (2, 3, 4, 5)
    last_dimensions = [-3, -2, -1]
    normalized_shape = to_tuple(get_list_from_index(shape, last_dimensions))
    x = get_a_sample_tensor(shape)
    x_bn = nn.LayerNorm(normalized_shape)(x)
    x_bn_check = get_tensor_norm(x, last_dimensions)
    assert_tensor_close(x_bn, x_bn_check, abs_tol=1e-6)


# per sample per channel
# 一定保留 C，如果有 N，也保留 N，收缩几维，就有多少对 (γ, β)
# (C, L) -> (C, 1)
# nlp
# (N, L, C) -> (N, L, 1)
# (C, H, W) -> (C, 1, 1)
# cv
# (N, C, H, W) -> (N, C, 1, 1)
def check_instance_norm():
    # (C, L) -> (C, 1)
    c_index = 0
    shape = (2, 3)
    c = shape[c_index]
    x = get_a_sample_tensor(shape)
    x_bn = nn.InstanceNorm1d(c)(x)
    x_bn_check = get_tensor_norm(x, 1)
    assert_tensor_close(x_bn, x_bn_check)

    # (N, L, C) -> (N, L, 1)
    c_index = 1
    shape = (2, 3, 4)
    c = shape[c_index]
    x = get_a_sample_tensor(shape)
    x_bn = nn.InstanceNorm1d(c)(x)
    x_bn_check = get_tensor_norm(x, 2)
    assert_tensor_close(x_bn, x_bn_check)

    # (C, H, W) -> (C, 1, 1)
    c_index = 0
    shape = (2, 3, 4)
    c = shape[c_index]
    x = get_a_sample_tensor(shape)
    x_bn = nn.InstanceNorm2d(c)(x)
    x_bn_check = get_tensor_norm(x, (1, 2))
    assert_tensor_close(x_bn, x_bn_check, abs_tol=1e-6)

    # (N, C, H, W) -> (N, C, 1, 1)
    c_index = 1
    shape = (2, 3, 4, 5)
    c = shape[c_index]
    x = get_a_sample_tensor(shape)
    x_bn = nn.InstanceNorm2d(c)(x)
    x_bn_check = get_tensor_norm(x, (2, 3))
    assert_tensor_close(x_bn, x_bn_check, abs_tol=1e-6)


# per sample per group
# group_norm 是 layer_norm 的特殊情况
# 只对指定的后几维做收缩
# normalized_shape = 最后几个维度，收缩几维，就有多少对 (γ, β)
# num_channels must be divisible by num_groups
# num_groups 必须能均分 C
# (N, C) -> num_groups * (N, 1)
# (N, C, L) -> num_groups * (N, 1, 1)
# (N, C, H, W) -> num_groups * (N, 1, 1, 1)

# NLP Example
# embedding = torch.randn(batch, sentence_length, embedding_dim)
# layer_norm = nn.LayerNorm(embedding_dim)
# layer_norm(embedding)

# Image Example
# N, C, H, W = 20, 5, 10, 10
# input = torch.randn(N, C, H, W)
# layer_norm = nn.LayerNorm([C, H, W])
def check_group_norm():
    c_index = 1

    for num_groups in (1, 2, 4):
        # (N, C) -> num_groups * (N, 1)
        shape = (2, 4)
        c = shape[c_index]
        x = get_a_sample_tensor(shape)
        x_bn = nn.GroupNorm(num_groups, c)(x)
        x_bn_check_list = list()
        for sub_x in torch.split(x, c // num_groups, dim=c_index):
            sub_x_bn_check = get_tensor_norm(sub_x, 1)
            x_bn_check_list.append(sub_x_bn_check)
        # cat 是 split 的逆操作，这里用于 split 一样的 dim
        x_bn_check = torch.cat(x_bn_check_list, dim=c_index)
        assert_tensor_close(x_bn, x_bn_check)

        # (N, C, L) -> num_groups * (N, 1, 1)
        shape = (2, 4, 5)
        c = shape[c_index]
        x = get_a_sample_tensor(shape)
        x_bn = nn.GroupNorm(num_groups, c)(x)
        x_bn_check_list = list()
        for sub_x in torch.split(x, c // num_groups, dim=c_index):
            sub_x_bn_check = get_tensor_norm(sub_x, (1, 2))
            x_bn_check_list.append(sub_x_bn_check)
        # cat 是 split 的逆操作，这里用于 split 一样的 dim
        x_bn_check = torch.cat(x_bn_check_list, dim=c_index)
        assert_tensor_close(x_bn, x_bn_check)

        # (N, C, H, W) -> num_groups * (N, 1, 1, 1)
        shape = (2, 4, 5, 6)
        c = shape[c_index]
        x = get_a_sample_tensor(shape)
        x_bn = nn.GroupNorm(num_groups, c)(x)
        x_bn_check_list = list()
        for sub_x in torch.split(x, c // num_groups, dim=c_index):
            sub_x_bn_check = get_tensor_norm(sub_x, (1, 2, 3))
            x_bn_check_list.append(sub_x_bn_check)
        # cat 是 split 的逆操作，这里用于 split 一样的 dim
        x_bn_check = torch.cat(x_bn_check_list, dim=c_index)
        assert_tensor_close(x_bn, x_bn_check, abs_tol=1E-6)


# https://pytorch.org/docs/2.2/generated/torch.nn.utils.parametrizations.weight_norm.html#torch.nn.utils.parametrizations.weight_norm
# w = g * v / |v|
# 将 weight 权重矩阵拆成两个部分，一个是幅值 g，另外一个是方向 v / |v|
# v = w, g = |w| 所以 g * v / |v| = w
# 其中，|w| = w.norm(dim=1, keepdim=True))
def check_weight_norm():
    shape = (3, 5)
    x = get_a_sample_tensor(shape)

    linear = nn.Linear(5, 7, bias=False)
    y = linear(x)
    assert_tensor_shape_equal(y, (3, 7))

    weight = linear.weight
    assert_tensor_shape_equal(weight, (7, 5))
    assert_tensor_shape_equal(weight.T, (5, 7))
    y_check = torch.matmul(x, weight.T)
    assert_tensor_equal(y, y_check)

    wn_linear = torch.nn.utils.parametrizations.weight_norm(linear)
    # weight 本身是不变的，只是每次进行 forward 之前会利用下面的 g 和 v 来计算
    assert_tensor_equal(weight, wn_linear.weight)
    wn_y = wn_linear(x)
    assert_tensor_equal(y, wn_y)

    # 每个 x 是一个 5D vector，因此每个 w 是一个 5D vector，因为输出层是 7， 因此需要有 7 个 w 组成一个 W
    # 这里矩阵范数 |W| 是表示将每个 w 进行归一化 (L2 范数)，这样 W @ W = 的对角线都是 1
    weight_norm = weight.norm(dim=1, keepdim=True)
    # 另外一种计算矩阵范数的方法
    assert_tensor_equal(weight_norm,
                        torch.tensor([get_vector_norm(weight[i, :]) for i in torch.arange(weight.shape[0])])
                        .unsqueeze(-1))
    assert_tensor_shape_equal(weight_norm, (7, 1))
    weight_direction = weight / weight_norm

    assert_tensor_close(weight_direction.norm(dim=-1), [1.0] * 7)
    assert_tensor_close((weight_direction @ weight_direction.T).diagonal().sum(), 7.0)

    # # 利用矩阵范数归一化，使得 weight_v 表示方向，矩阵范数 = 矩阵各项元素平方和再开根号，注意这里用的并不是这个范数
    weight_v = wn_linear.parametrizations.weight.original1
    weight_g = wn_linear.parametrizations.weight.original0
    assert_tensor_shape_equal(weight_g, (7, 1))
    assert_tensor_shape_equal(weight_v, (7, 5))

    assert_tensor_equal(weight_norm, weight_g)
    assert_tensor_equal(weight, weight_v)


def check_gpu(_with_speed=False):
    # 2.2.1+cu121
    # print(torch.__version__)

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled

    if _with_speed:
        dimension = 5000

        # i9-9900K
        # spent 111.40064930915833
        # i9-14900KF
        # spent 40.08144783973694
        # device = torch.device("cpu")

        # 2080Ti
        # spent 4.195726633071899
        # 4090
        # spent 2.9713356494903564
        device = torch.device("cuda")

        x = torch.rand((dimension, dimension), dtype=torch.float32)
        y = torch.rand((dimension, dimension), dtype=torch.float32)

        x = x.to(device)
        y = y.to(device)

        start_time = time.time()
        for i in range(10000):
            # noinspection PyUnusedLocal
            z = x * y
        end_time = time.time()

        # 总显存 (GB):      2.0
        # torch 显存 (GB):  0.4
        # tensor 显存 (GB): 0.3
        print_gpu_memory_summary()

        print("spent {}".format(end_time - start_time))


# 检查 half 的用法，其实就是转化为 float 16
def check_half():
    float_64 = torch.tensor([3.1415926], dtype=torch.float64)
    float_32 = torch.tensor([3.1415926], dtype=torch.float32)
    float_16 = torch.tensor([3.1415926], dtype=torch.float16)

    # tensor([3.1416], dtype=torch.float64)
    # tensor([3.1406], dtype=torch.float16)
    # tensor([3.1406], dtype=torch.float16)
    print(float_64)
    print(float_64.half())
    print(float_64.half().half())

    # tensor([3.1416])
    # tensor([3.1406], dtype=torch.float16)
    # tensor([3.1406], dtype=torch.float16)
    print(float_32)
    print(float_32.half())
    print(float_32.half().half())

    # tensor([3.1406], dtype=torch.float16)
    # tensor([3.1406], dtype=torch.float16)
    # tensor([3.1406], dtype=torch.float16)
    print(float_16)
    print(float_16.half())
    print(float_16.half().half())


@func_timer(arg=True)
def check_chatglm3():
    from transformers import AutoModel, AutoTokenizer
    # trust_remote_code 表示相信本地的代码，而不是表示同意下载远程代码，不要混淆
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CHATGLM3_6B_model_dir,
                                              trust_remote_code=True)
    # <class 'transformers_modules.chatglm3-6b.tokenization_chatglm.ChatGLMTokenizer'>
    print(type(tokenizer))
    # ['SPECIAL_TOKENS_ATTRIBUTES', 'add_special_tokens', 'add_tokens', 'added_tokens_decoder', 'added_tokens_encoder',
    # 'additional_special_tokens', 'additional_special_tokens_ids', 'all_special_ids', 'all_special_tokens',
    # 'all_special_tokens_extended', 'apply_chat_template', 'as_target_tokenizer', 'batch_decode', 'batch_encode_plus',
    # 'bos_token', 'bos_token_id', 'build_chat_input', 'build_inputs_with_special_tokens', 'build_single_message',
    # 'chat_template', 'clean_up_tokenization', 'clean_up_tokenization_spaces', 'cls_token', 'cls_token_id',
    # 'convert_added_tokens', 'convert_ids_to_tokens', 'convert_tokens_to_ids', 'convert_tokens_to_string',
    # 'create_token_type_ids_from_sequences', 'decode', 'default_chat_template', 'deprecation_warnings',
    # 'encode', 'encode_plus', 'eos_token', 'eos_token_id', 'from_pretrained', 'get_added_vocab', 'get_command',
    # 'get_prefix_tokens', 'get_special_tokens_mask', 'get_vocab', 'init_inputs', 'init_kwargs', 'is_fast',
    # 'mask_token', 'mask_token_id', 'max_len_sentences_pair', 'max_len_single_sentence', 'max_model_input_sizes',
    # 'model_input_names', 'model_max_length', 'name', 'name_or_path', 'num_special_tokens_to_add', 'pad',
    # 'pad_token', 'pad_token_id', 'pad_token_type_id', 'padding_side', 'prepare_for_model', 'prepare_for_tokenization',
    # 'prepare_seq2seq_batch', 'pretrained_init_configuration', 'pretrained_vocab_files_map', 'push_to_hub',
    # 'register_for_auto_class', 'sanitize_special_tokens', 'save_pretrained', 'save_vocabulary', 'sep_token',
    # 'sep_token_id', 'slow_tokenizer_class', 'special_tokens', 'special_tokens_map', 'special_tokens_map_extended',
    # 'split_special_tokens', 'tokenize', 'tokenizer', 'tokens_trie', 'truncate_sequences', 'truncation_side',
    # 'unk_token', 'unk_token_id', 'verbose', 'vocab_file', 'vocab_files_names', 'vocab_size']
    print_dir(tokenizer)

    dictionary = tokenizer.get_vocab()
    # <class 'dict'> 64796 True
    # 字典
    print(type(dictionary), len(dictionary), "月光" in dictionary)

    # encode 就是 encode_plus 的一部分
    # return self.encode_plus()["input_ids"]
    # [64790, 64792, 34211, 51225, 34886, 30930]
    # print(tokenizer.encode('我爱我老婆.'))

    # {'input_ids': [0, 0, 64790, 64792, 34211, 51225, 34886, 30930, 34211, 34886, 54532, 55266, 54678, 30930, 2],
    # 'token_type_ids': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 'special_tokens_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 'attention_mask': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # 'position_ids': [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
    # 支持传一句话或者两句话，如每句话的开头有 "_"
    # 如果要想批量编码，调用 batch_encode_plus，会增加一个维度，表示 batch
    sen_code = tokenizer.encode_plus('我爱我老婆.', '我老婆是陈平.', truncation=True, max_length=15,
                                     padding="max_length", return_token_type_ids=True, return_special_tokens_mask=True)
    print(sen_code)
    # ['', '', '[gMASK]', 'sop', '▁我', '爱我', '老婆', '.', '▁我', '老婆', '是', '陈', '平', '.', '']
    print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

    sen_code = tokenizer.encode_plus('你说什么.', '这个课程太难学了.')
    print(sen_code)
    # ['[gMASK]', 'sop', '▁你', '说什么', '.', '▁这个', '课程', '太难', '学了', '.', '']
    print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

    # 通过查看 config.json，torch_dtype = float16"，因此这里用不用 half 都可以
    model = AutoModel.from_pretrained(CHATGLM3_6B_model_dir, trust_remote_code=True).cuda()
    # <class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>
    print(type(model))
    # ['T_destination', 'active_adapter', 'active_adapters', 'add_adapter', 'add_memory_hooks', 'add_model_tags',
    # 'add_module', 'apply', 'assisted_decoding', 'base_model', 'base_model_prefix', 'beam_sample', 'beam_search',
    # 'bfloat16', 'buffers', 'call_super_init', 'can_generate', 'chat', 'children', 'compile',
    # 'compute_transition_scores', 'config', 'config_class', 'constrained_beam_search', 'contrastive_search', 'cpu',
    # 'create_extended_attention_mask_for_decoder', 'cuda', 'device', 'disable_adapters', 'disable_input_require_grads',
    # 'double', 'dtype', 'dummy_inputs', 'dump_patches', 'enable_adapters', 'enable_input_require_grads',
    # , 'eval', 'extra_repr', 'float', 'floating_point_ops', 'forward', 'framework', 'from_pretrained', 'generate',
    # 'generation_config', 'get_adapter_state_dict', 'get_buffer', 'get_extended_attention_mask', 'get_extra_state',
    # 'get_head_mask', 'get_input_embeddings', 'get_masks', 'get_memory_footprint', 'get_output_embeddings',
    # 'get_parameter', 'get_position_embeddings', 'get_position_ids', 'get_submodule', 'gradient_checkpointing_disable',
    # 'gradient_checkpointing_enable', 'greedy_search', 'group_beam_search', 'half', 'init_weights',
    # 'invert_attention_mask', 'ipu', 'is_gradient_checkpointing', 'is_parallelizable', 'load_adapter',
    # 'load_state_dict', 'main_input_name', 'max_sequence_length', 'model_tags', 'modules', 'name_or_path',
    # 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'num_parameters', 'parameters',
    # 'post_init', 'prepare_inputs_for_generation', 'process_response', 'prune_heads', 'push_to_hub', 'quantize',
    # 'quantized', 'register_backward_hook', 'register_buffer', 'register_for_auto_class', 'register_forward_hook',
    # 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook',
    # 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'register_state_dict_pre_hook',
    # 'requires_grad_', 'reset_memory_hooks_state', 'resize_position_embeddings', 'resize_token_embeddings',
    # 'retrieve_modules_from_names', 'reverse_bettertransformer', 'sample', 'save_pretrained', 'set_adapter',
    # 'set_extra_state', 'set_input_embeddings', 'share_memory', 'state_dict', 'stream_chat', 'stream_generate',
    # 'supports_gradient_checkpointing', 'tie_weights', 'to', 'to_bettertransformer', 'to_empty', 'train', 'training',
    # 'transformer', 'type', 'warn_if_padding_and_no_attention_mask', 'warnings_issued', 'xpu', 'zero_grad']
    print_dir(model)

    # model = AutoModel.from_pretrained(CHATGLM3_6B_model_dir, trust_remote_code=True).half().cuda()
    total_parameters = model.num_parameters()
    # 总显存 (GB):      13.22
    # torch 显存 (GB):  11.66
    # tensor 显存 (GB): 11.66
    print_gpu_memory_summary()

    # 参数量：6243584000，占用显存: 11.63 GB
    print(F"参数量：{total_parameters}，占用显存: {round(total_parameters * 2 / 1024 ** 3, 2)} GB")

    # ================================================================================
    # Layer (type:depth-idx)                                  Param #
    # ================================================================================
    # ChatGLMForConditionalGeneration                         --
    # ├─ChatGLMModel: 1-1                                     --
    # │    └─Embedding: 2-1                                   --
    # │    │    └─Embedding: 3-1                              266,338,304
    # │    └─RotaryEmbedding: 2-2                             --
    # │    └─GLMTransformer: 2-3                              --
    # │    │    └─ModuleList: 3-2                             5,710,903,296
    # │    │    └─RMSNorm: 3-3                                4,096
    # │    └─Linear: 2-4                                      266,338,304
    # ================================================================================
    # Total params: 6,243,584,000
    # Trainable params: 6,243,584,000
    # Non-trainable params: 0
    # ================================================================================
    # 注意，需要给 input 才能知道整个的参数量
    summary(model)

    # ChatGLMForConditionalGeneration(
    #   (transformer): ChatGLMModel(
    #     (embedding): Embedding(
    #       (word_embeddings): Embedding(65024, 4096)
    #     )
    #     (rotary_pos_emb): RotaryEmbedding()
    #     (encoder): GLMTransformer(
    #       (layers): ModuleList(
    #         (0-27): 28 x GLMBlock(
    #           (input_layernorm): RMSNorm()
    #           (self_attention): SelfAttention(
    #             (query_key_value): Linear(in_features=4096, out_features=4608, bias=True)
    #             (core_attention): CoreAttention(
    #               (attention_dropout): Dropout(p=0.0, inplace=False)
    #             )
    #             (dense): Linear(in_features=4096, out_features=4096, bias=False)
    #           )
    #           (post_attention_layernorm): RMSNorm()
    #           (mlp): MLP(
    #             (dense_h_to_4h): Linear(in_features=4096, out_features=27392, bias=False)
    #             (dense_4h_to_h): Linear(in_features=13696, out_features=4096, bias=False)
    #           )
    #         )
    #       )
    #       (final_layernorm): RMSNorm()
    #     )
    #     (output_layer): Linear(in_features=4096, out_features=65024, bias=False)
    #   )
    # )
    print(model)

    # =========================================================================================================
    # Layer (type:depth-idx)                                  Output Shape              Param #
    # =========================================================================================================
    # ChatGLMForConditionalGeneration                         [512, 16, 2, 128]         --
    # ├─ChatGLMModel: 1-1                                     [512, 16, 2, 128]         --
    # │    └─Embedding: 2-1                                   [512, 16, 4096]           --
    # │    │    └─Embedding: 3-1                              [16, 512, 4096]           266,338,304
    # │    └─RotaryEmbedding: 2-2                             [8192, 32, 2]             --
    # │    └─GLMTransformer: 2-3                              [512, 16, 4096]           --
    # │    │    └─ModuleList: 3-2                             --                        5,710,903,296
    # │    │    └─RMSNorm: 3-3                                [512, 16, 4096]           4,096
    # │    └─Linear: 2-4                                      [512, 16, 65024]          266,338,304
    # =========================================================================================================
    # Total params: 6,243,584,000
    # Trainable params: 6,243,584,000
    # Non-trainable params: 0
    # Total mult-adds (Units.TERABYTES): 3.06
    # =========================================================================================================
    # Input size (MB): 0.03
    # Forward/backward pass size (MB): 46791.66
    # Params size (MB): 12487.17
    # Estimated Total Size (MB): 59278.86
    # =========================================================================================================
    summary(model, input_size=(16, 512), dtypes=[torch.int])
    model = model.eval()

    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    # print_history_message_list(history)

    # 1. 尝试放松身心，如深呼吸、冥想或温和的瑜伽。
    # 2. 避免刺激性食物和饮料，如咖啡、茶和巧克力。
    # 3. 保持规律的睡眠时间表。
    # 4. 尝试舒适的环境，如调暗灯光、使用白噪音或舒适的床垫。
    # 5. 避免在晚上过度使用电子设备，如手机、平板电脑和电视。
    # 6. 保持适度的运动，如散步、瑜伽或伸展运动。
    # 7. 如果需要，可以考虑采用放松技巧，如渐进性肌肉松弛或呼吸练习。
    # 8. 睡前适当限制使用兴奋剂，如尼古丁和酒精。
    # 9. 睡前尝试冥想或深度放松练习。
    # 10. 如有必要，可咨询医生或专业心理健康专家。
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办，回复字数不要超过 100 个", history=history)
    print(response)
    # print_history_message_list(history)

    # 1. 尝试调整咖啡因摄入量，控制在一日 limit 内。
    # 2. 尝试其他非咖啡因的提神饮料，如茶、果汁或苏打水。
    # 3. 考虑采用放松技巧，如冥想或深度放松练习。
    # 4. 增加白天休息时间，如小憩或午睡。
    # 5. 调整饮食结构，增加易消化的食物，如坚果、全麦面包或香蕉。
    # 6. 尝试进行有氧运动，如跑步或游泳。
    # 7. 保持良好的睡眠时间表，尽量在同一时间入睡和起床。
    # 8. 避免在睡前过度使用电子设备，如手机、平板电脑和电视。
    # 9. 睡前适当限制咖啡因摄入，如减少咖啡或茶摄入量。
    # 10. 如有必要，可咨询医生或专业心理健康专家。
    response, history = model.chat(tokenizer, "但我工作的原因必须喝咖啡，回复字数不要超过 100 个", history=history)
    print(response)
    # print_history_message_list(history)

    # 我明白您的工作原因需要喝咖啡来保持清醒和提高工作效率。咖啡因是一种兴奋剂，可以增加警觉性和注意力，帮助您更好地应对日常任务。当然，适量饮用咖啡对大多数人来说是安全的，但请注意不要过量摄入咖啡因，以免出现不良反应。
    # 历史对话需要通过传入 history 来引入，否则模型记不住上下文
    response, history = model.chat(tokenizer, "但我工作的原因必须喝咖啡，回复字数不要超过 100 个", history=[])
    print(response)
    # print_history_message_list(history)


def main():
    # 2.2.0+cu121
    print(torch.__version__)
    check_gpu(True)
    check_mul()
    check_mean_op()
    check_std_op()
    other_simple()
    check_batch_norm()
    check_layer_norm()
    check_instance_norm()
    check_group_norm()
    check_weight_norm()
    check_half()
    check_chatglm3()


if __name__ == '__main__':
    main()
