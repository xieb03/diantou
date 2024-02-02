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
    # 2.2.0+cu118
    # print(torch.__version__)

    assert torch.cuda.is_available()

    if _with_speed:
        # dimension = 5000
        # spent 111.40064930915833
        # i9-9900K
        # device = torch.device("cpu")
        # spent 4.195726633071899
        # 2080Ti
        device = torch.device("cuda")

        dimension = 5000
        x = torch.rand((dimension, dimension), dtype=torch.float32)
        y = torch.rand((dimension, dimension), dtype=torch.float32)

        x = x.to(device)
        y = y.to(device)

        start_time = time.time()
        for i in range(10000):
            z = x * y
        end_time = time.time()

        print("spent {}".format(end_time - start_time))


def main():
    # 2.2.0+cu118
    print(torch.__version__)
    check_gpu()
    check_mul()
    check_mean_op()
    check_std_op()
    other_simple()
    check_batch_norm()
    check_layer_norm()
    check_instance_norm()
    check_group_norm()
    check_weight_norm()


if __name__ == '__main__':
    main()
