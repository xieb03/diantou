# noinspection PyUnresolvedReferences
import tiktoken
import torch
# noinspection PyUnresolvedReferences
from torch import nn

from util import *


# https://zhuanlan.zhihu.com/p/629526120
# 在多进程数据加载算法中，DataLoader将根据Randomness in multi-process data loading algorithm对worker进行重新初始化种子。使用worker_init_fn()来保持可再现性:
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     numpy.random.seed(worker_seed)
#     random.seed(worker_seed)
#
# DataLoader(
#     xxx,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker,
# )
# noinspection PyUnusedLocal


# 查看模型的 device
def check_model_device(_model):
    print(next(_model.parameters()).device)


# 一个 model 是否在 gpu 上
def is_model_on_gpu(_model):
    return "cuda" in str(next(_model.parameters()).device)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 将一个 dict 的 所有 value 都放到 gpu 中，注意是原位修改
def change_dict_value_to_gpu(_dict):
    for k, v in _dict.items():
        assert isinstance(v, torch.Tensor)
        _dict[k] = v.cuda()


# 获取现存的整体情况
# 总显存 (GB):      2.0
# torch 显存 (GB):  0.4
# tensor 显存 (GB): 0.3
def print_gpu_memory_summary(_digit=2):
    print("total  gpu memory: ", round(get_total_gpu_memory() / 1024 / 1024 / 1024, _digit), "G")
    # 其它的一些开销，例如 torch 本身占据的缓存
    print("torch  gpu memory: ", round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, _digit), "G")
    # 仅仅是 tensor 占用的
    print("tensor gpu memory: ", round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, _digit), "G")


# 获得 dtype 对应的 byte 数
def get_dtype_byte_count(_dtype):
    # 半精度
    if _dtype in (torch.float16, torch.half):
        return 2
    # 浮点数
    elif _dtype in (torch.float32, torch.float):
        return 4
    # 双精度
    elif _dtype in (torch.float64, torch.double):
        return 8
    # 带符号/无符号 8 位整数
    elif _dtype in (torch.int8, torch.uint8):
        return 1
    # 带符号 16 位整数
    elif _dtype in (torch.int16, torch.short):
        return 2
    # 带符号 32 位整数
    elif _dtype in (torch.int32, torch.int):
        return 4
    # 带符号 64 位整数
    elif _dtype in (torch.int64, torch.long):
        return 8
    else:
        raise ValueError(F"暂时不支持 {_dtype} 数据类型.")


# 打印 model 的参数情况
# 与 model.print_trainable_parameters() 相比较
def print_model_parameter_summary(_model):
    # 并不是所有的 model 都有 num_parameters 属性，但一定有 parameters() 方法
    total_parameter_count, trainable_params_count, memory = list_model_parameter_summary(_model, _print=False)
    if hasattr(_model, "num_parameters"):
        # num_parameters 中不包含 buffer，因此可能会少
        assert total_parameter_count >= _model.num_parameters(), F"{total_parameter_count} < {_model.num_parameters()}"

    if hasattr(_model, "dtype"):
        print(
            F"model ({type(_model)}) has {total_parameter_count} parameters, {trainable_params_count} ({trainable_params_count / total_parameter_count:.2%}) are trainable, the dtype is {_model.dtype}，占 {round(memory, 2)}G 显存.")
    else:
        print(
            F"model ({type(_model)}) has {total_parameter_count} parameters, {trainable_params_count} ({trainable_params_count / total_parameter_count:.2%}) are trainable, 占 {round(memory, 2)}G 显存.")


# 列出 parameter 情况，注意区分 trainable parameter, distrainable parameter, distrainable buffers:
def list_model_parameter_summary(_model, _print=True):
    total_parameter_count = 0
    trainable_params_count = 0
    memory = 0

    if _print:
        print("trainable parameter:")
    for i, (name, parameter) in enumerate(_model.named_parameters()):
        if parameter.requires_grad:
            total_parameter_count += parameter.numel()
            trainable_params_count += parameter.numel()
            memory += parameter.numel() * get_dtype_byte_count(parameter.dtype) / 1024 ** 3
            if _print:
                print(f"\t{i}: {name}, {parameter.dtype}, {parameter.size()}, {parameter.requires_grad}")

    if _print:
        print("distrainable parameter:")
    for i, (name, parameter) in enumerate(_model.named_parameters()):
        if not parameter.requires_grad:
            total_parameter_count += parameter.numel()
            memory += parameter.numel() * get_dtype_byte_count(parameter.dtype) / 1024 ** 3
            if _print:
                print(f"\t{i}: {name}, {parameter.dtype}, {parameter.size()}, {parameter.requires_grad}")

    if _print:
        print("distrainable buffers:")
    for i, (name, buffer) in enumerate(_model.named_buffers()):
        assert not buffer.requires_grad
        total_parameter_count += buffer.numel()
        memory += buffer.numel() * get_dtype_byte_count(buffer.dtype) / 1024 ** 3
        if _print:
            print(f"\t{i}: {name}, {buffer.dtype}, {buffer.size()}, {buffer.requires_grad}")

    if _print:
        if hasattr(_model, "dtype"):
            print(
                F"model ({type(_model)}) has {total_parameter_count} parameters, {trainable_params_count} ({trainable_params_count / total_parameter_count:.2%}) are trainable, the dtype is {_model.dtype}，占 {round(memory, 2)}G 显存.")
        else:
            print(
                F"model ({type(_model)}) has {total_parameter_count} parameters, {trainable_params_count} ({trainable_params_count / total_parameter_count:.2%}) are trainable, 占 {round(memory, 2)}G 显存.")
    return total_parameter_count, trainable_params_count, memory


# 获取显卡的整体情况
# 可用 GPU 数量: 1
# GPU 0 的详细信息:
# 名称: NVIDIA GeForce RTX 4090
# 计算能力: 8.9
# 内存总量 (GB): 24.0
def print_gpu_device_summary():
    num_devices = torch.cuda.device_count()
    print("可用 GPU 数量:", num_devices)

    # 遍历所有可用的 GPU 设备并打印详细信息
    for i in range(num_devices):
        device = torch.cuda.get_device_properties(i)
        print(F"GPU {i} 的详细信息:")
        print("名称:", device.name)
        print("计算能力:", f"{device.major}.{device.minor}")
        print("显存总量 (GB):", round(device.total_memory / (1024 ** 3), 1))


# 获取当前显卡所有已用的缓存，通过 nvidia-smi 获取
# 注意返回的是 MB，乘以 1024^2 转化为 byte
def get_total_gpu_memory():
    return (1024 * 1024 *
            int(os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").read().split("\n")[0]))


# 获取一个 sample
def get_a_sample_tensor(_shape, _random=False, _dtype=torch.float):
    _shape = to_tuple(_shape)
    if _random:
        return torch.randn(size=_shape, dtype=_dtype)

    count = 1
    for value in _shape:
        count *= value
    return torch.arange(count, dtype=_dtype).reshape(_shape)


# 统一转化为 tensor 来比较
def assert_tensor_equal(a, b, _force=False):
    # torch.equal 的两个输入必须都是 Tensor，直接返回 True or False
    # To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
    # or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
    # assert torch.equal(torch.tensor(a), torch.tensor(b)), F"{a} != {b}"
    x, y = a, b
    if not isinstance(a, torch.Tensor):
        x = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        y = torch.tensor(b)
    assert torch.equal(x, y), F"{a} != {b}"


# 统一转化为 tensor 来比较
def assert_tensor_close(a, b, rel_tol=1e-05, abs_tol=1e-08):
    # torch.isclose 的两个输入必须都是 Tensor，返回 dtype = bool 的 tensor
    x, y = a, b
    if not isinstance(a, torch.Tensor):
        x = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        y = torch.tensor(b)
    assert x.shape == y.shape, F"{x.shape} not close {y.shape}"
    assert torch.isclose(x, y, rel_tol, abs_tol).all(), (
        "{} not close {}, rel_err = {}, abs_err = {}".format(a, b, *get_tensor_rel_abs_error(a, b)))


# 获得 rel_error 和 abs_error，都是全局最大的，且取过绝对值
def get_tensor_rel_abs_error(a, b):
    assert_tensor_shape_equal(a, b)

    abs_error = torch.max(torch.abs(a - b))
    rel_error = torch.max(abs_error / torch.minimum(torch.abs(a), torch.abs(b)))
    return rel_error, abs_error


# 比较一个 tensor 的形状，注意，必须用 tuple 而不能是 list
# 如果是 torch.Size([])，需要传入 tuple() 进行比较，当然也可以传入 torch.Size([])
# 也可以传入两个 tensor 直接比较形状
def assert_tensor_shape_equal(_tensor: torch.Tensor, _shape_or_tensor):
    if isinstance(_shape_or_tensor, torch.Tensor):
        # 注意 shape 是 torch.Size 不是 torch.Shape
        assert_tensor_equal(_tensor.shape, _shape_or_tensor.shape)
    else:
        assert_tensor_equal(_tensor.shape, to_tuple(_shape_or_tensor))


# 获得向量 vector 的 p 范数，默认 p = 1
def get_vector_norm(_vector: torch.Tensor, _p=2):
    # 必须是 vector
    assert _vector.dim() == 1, _vector.dim
    return torch.linalg.vector_norm(_vector, ord=_p)


# 求正则化结果
def get_tensor_norm(_x: torch.Tensor, _dim, _keepdim=True, _unbiased=False, _eps=1E-5):
    return (_x - _x.mean(dim=_dim, keepdim=_keepdim)) / torch.sqrt(
        (_x.var(dim=_dim, keepdim=_keepdim, unbiased=_unbiased) + _eps))


def main():
    pass


if __name__ == '__main__':
    main()
