import dataclasses as dc
import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import jieba
import ruamel.yaml as yaml
from datasets import Dataset, DatasetDict, NamedSplit, Split
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from transformers import (
    AutoModelForCausalLM,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

sys.path.append("../")
from project_utils import *

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


# 解析路径
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


# 将原始数据转化为 lora 需要的 conversations 格式
def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path):
        _mkdir(out_file.parent)
        count = 0
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    count += 1
                    dct = json.loads(line)
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
        return count

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / 'train.json'
        sample_count = _convert(train_file)
        print(F"{train_file} 共有 {sample_count} 行.")

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / 'dev.json'
        sample_count = _convert(dev_file)
        print(F"{dev_file} 共有 {sample_count} 行.")


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels


# noinspection PyUnresolvedReferences
def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


# 缓存，只会执行一次
@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


# dataclass是python3.7开始带有的新属性(类装饰器)，
# dataclass是指”一个带有默认值的可变namedtuple“，本质还是一个类，它的属性非特殊情况可以直接访问，类中有与属性相关的类方法。简单地说就是一个含有数据及其操作方法的类。
@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    # 原来是 default=...，会报错
    # mutable default <class 'transformers.training_args_seq2seq.Seq2SeqTrainingArguments'> for field training_args
    # is not allowed: use default_factory
    # 改为 default_factory=...
    # training_args: Seq2SeqTrainingArguments = dc.field(
    #     default_factory=Seq2SeqTrainingArguments(output_dir='./output')
    # )
    # 都从 kwargs 里面来
    training_args: Seq2SeqTrainingArguments
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    # noinspection PyArgumentList,PyTypeChecker,PyUnresolvedReferences
    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            # <class 'dict'>
            # print(type(data_config))
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            # <class 'dict'>
            # print(type(peft_config))
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


# noinspection PyTypeChecker
def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = datasets.load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> model has {total_params / 1e6}M params\n")


# noinspection PyUnresolvedReferences
def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


# noinspection PyUnresolvedReferences
def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            # if train with cpu, cast all params to fp32 instead of trainable ones.
            param.data = param.data.to(torch.float32)


# noinspection PyUnresolvedReferences
def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
            print_model_parameter_summary(model)
        elif peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            # model (<class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>)
            # has 6243584000 parameters, 6243584000 (100.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
            print_model_parameter_summary(model)
            model = get_peft_model(model, peft_config)
            # model (<class 'peft.peft_model.PeftModelForCausalLM'>)
            # has 6245533696 parameters, 1949696 (0.03%) are trainable, the dtype is torch.float16，占 11.63G 显存.
            # 可以看到 peft 在原始 model 基础上做了一些调整
            print_model_parameter_summary(model)
            # trainable params: 1,949,696 || all params: 6,245,533,696 || trainable%: 0.031217444255383614
            model.print_trainable_parameters()
        else:
            raise ValueError(f"peft_type 目前只支持 PREFIX_TUNING 和 LORA，不支持 {peft_config.peft_type.name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    print_model_size(model)
    return tokenizer, model


# noinspection PyTypeChecker
def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}


# 如果 batch_size 设置的比较大，就会出现下面的两个 UserWarning，特别是第 2 个，不能控制随机数生成固定结果，感觉是和 GPU 共享内存有关或者说显存到达一定程度会触发某种 cudnn 优化，而如果 batch_size 较小就没有这个问题
# 但如果将 warn_only 改成 False 也可以运行被可以生成随机结果，可能是 warn_only=False 会强制找到确定性的方法，如果找不到再报错，而 warn_only=True 只是出现警告而已
# C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:231: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
#   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
# D:\Users\admin\anaconda3\Lib\site-packages\torch\autograd\__init__.py:266: UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False). (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\transformers\cuda\attention_backward.cu:451.)
#   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
def fine_tune(
        data_dir: str,
        # A string that specifies the model id of a pretrained model configuration hosted on huggingface.co,
        # or a path to a directory containing a model configuration file.
        model_dir: str,
        config_file: str,
        # If entered as yes, automatically use the latest save checkpoint.
        # If it is a numerical example 12 15, use the corresponding save checkpoint.
        # If the input is None, restart training
        # 不再用 typer.Argument 格式，因为并不是从命令行输出的，并不会触发格式转换，例如报
        # 'main' get something wrong: 'ArgumentInfo' object has no attribute 'upper'
        auto_resume_from_checkpoint: str = None
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # Setting num_proc from 16 back to 1 for the train split to disable multiprocessing as it only contains one shard.
    # Generating train split: 114599 examples [00:00, 1416197.35 examples/s]
    # Map (num_proc=16): 100%|██████████| 114599/114599 [00:09<00:00, 11696.96 examples/s]
    # train_dataset: Dataset({
    #     features: ['input_ids', 'labels'],
    #     num_rows: 114599
    # })
    print('train_dataset:', train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # Setting num_proc from 16 back to 1 for the validation split to disable multiprocessing
    # as it only contains one shard.
    # Generating validation split: 1070 examples [00:00, 178922.19 examples/s]
    # Map (num_proc=16): 100%|██████████| 1070/1070 [00:08<00:00, 123.54 examples/s]
    # val_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 1070
    # })
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # Setting num_proc from 16 back to 1 for the test split to disable multiprocessing as it only contains one shard.
    # Generating test split: 1070 examples [00:00, 214722.04 examples/s]
    # Map (num_proc=16): 100%|██████████| 1070/1070 [00:08<00:00, 126.17 examples/s]
    # test_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 1070
    # })
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    # _sanity_check(
    #     train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    # )

    # turn model to fp32
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)

    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    # noinspection PyUnresolvedReferences
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        # 注意用 min 做一下数量的保护，否则当数据集中样本数量较少时，会报错：
        # IndexError: Index 49 out of range for dataset of size 12.
        eval_dataset=val_dataset.select(list(range(min(val_dataset.num_rows, 50)))),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    # total  gpu memory:  12.8 G
    # torch  gpu memory:  11.67 G
    # tensor gpu memory:  11.66 G
    print_gpu_memory_summary()

    # Determine whether to continue training without breakpoints or if it is empty, then start training again directly
    if auto_resume_from_checkpoint is None or auto_resume_from_checkpoint == "":
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        # # 将 output dir 设置到 与训练样本一起的位置，而不在用配置文件中的 output_dir
        # 注意上面的 default_factory=Seq2SeqTrainingArguments(output_dir='./output') 也要改
        # output_dir = delete_end_path_separator(data_dir) + PATH_SEPARATOR + "output"

        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint
                      + ") has not been saved. Please search for the correct checkpoint in the model output directory")

    # # test stage
    # if test_dataset is not None:
    #     trainer.predict(test_dataset)


# 聊天记录示例
def get_sample_chat(_model, _tokenizer, _temperature=0.1):
    response, history = _model.chat(_tokenizer, "你好", history=[], temperature=_temperature)
    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    print(response)
    print("-" * 80)
    # print_history_message_list(history)

    # 1. 尝试放松身心，如冥想、深呼吸等。
    # 2. 避免咖啡因和尼古丁。
    # 3. 保持规律作息，尽量每天按时上床睡觉。
    # 4. 减少使用电子产品，尤其是睡前。
    # 5. 增加运动量，但避免剧烈运动。
    # 6. 睡前适当饮水，但避免过多。
    # 7. 尝试阅读或听轻音乐。
    # 8. 如有需要，可咨询专业人士。
    response, history = _model.chat(_tokenizer, "晚上睡不着应该怎么办，回复字数不要超过 100 个", history=history,
                                    temperature=_temperature)
    print(response)
    print("-" * 80)
    # print_history_message_list(history)

    # response, history = _model.chat(_tokenizer,
    #                                 "类型#裙*版型#显瘦*材质#网纱*风格#性感*裙型#百褶*裙下摆#压褶*裙长#连衣裙*裙衣门襟#拉链*裙衣门襟#套头*裙款式",
    #                                 temperature=_temperature)
    # # 这款连衣裙的版型设计，整体上采用了显瘦的款式，加上百褶的裙摆，整体上显得非常时尚，加上网纱的装饰，整体上显得非常性感，
    # # 加上拉链的装饰，整体上显得非常实用。加上超长版型的设计，加上超长裙摆的设计，整体上显得非常优雅，加上超长裙摆的设计，加上超长裙摆的设计，整体上显得非常优雅。
    # print(response)
    # # print_history_message_list(history)

    response, history = _model.chat(_tokenizer,
                                    "fly_to() 是什么含义，如果不知道就说不知道",
                                    temperature=_temperature)
    # 抱歉，我无法提供关于 "aw.fly_to()" 的详细信息，因为我无法确定您所提到的 "aw" 代表什么。如果您能提供更多上下文或信息，我将尽力帮助您。
    print(response)
    print("-" * 80)
    # print_history_message_list(history)

    with open("prompt/system_prompt.txt", "r", encoding="UTF8") as f:
        system_prompt = f.read()

    with open("prompt/user_prompt.txt", "r", encoding="UTF8") as f:
        user_prompt = f.read()

    response, history = _model.chat(_tokenizer,
                                    system_prompt, role="system",
                                    temperature=_temperature)
    print(response)
    print("-" * 80)
    # print_history_message_list(history)

    response, history = _model.chat(_tokenizer,
                                    user_prompt, history=history,
                                    temperature=_temperature)
    print(response)
    print("-" * 80)
    # print_history_message_list(history)

    response, history = _model.chat(_tokenizer,
                                    "向上升7米", history=history,
                                    temperature=_temperature)
    print(response)
    print("-" * 80)
    # print_history_message_list(history)


# 微调以前
# RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
# You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application.
# You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
# 通过 torch.use_deterministic_algorithms(True, warn_only=True)，可以禁止抛出异常而只是 warn，方便调试
# _temperature 必须要 > 0
def before_fine_tune(_temperature=0.01):
    # 单独为了 cumsum_cuda_kernel 关闭
    torch.use_deterministic_algorithms(False)

    model, tokenizer = load_chatglm_model_and_tokenizer()
    # model (<class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>)
    # has 6243584000 parameters, 6243584000 (100.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    print_model_parameter_summary(model)
    # total  gpu memory:  12.89 G
    # torch  gpu memory:  11.66 G
    # tensor gpu memory:  11.66 G
    print_gpu_memory_summary()

    get_sample_chat(model, tokenizer, _temperature)


# 微调以后
def after_fine_tune(_fine_tune_dir: str, _temperature=0.1):
    # 单独为了 cumsum_cuda_kernel 关闭
    torch.use_deterministic_algorithms(False)

    model, tokenizer = load_chatglm_model_and_tokenizer(_use_checkpoint=True, _checkpoint_path=_fine_tune_dir)
    # model (<class 'peft.peft_model.PeftModelForCausalLM'>) has 6245533696 parameters,
    # 0 (0.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    print_model_parameter_summary(model)
    # total  gpu memory:  12.76 G
    # torch  gpu memory:  11.67 G
    # tensor gpu memory:  11.66 G
    print_gpu_memory_summary()

    get_sample_chat(model, tokenizer, _temperature)


# chatGLM 自带的微调例子
def fine_tune_using_advertise(_save_dir, _model_dir=CHATGLM3_6B_model_dir, _temperature=0.1,
                              _config_file="./finetune_configs/lora.yaml", _with_fine_tune=False,
                              _checkpoint=r"./output/checkpoint-250"):
    # tensorflow sed random seed fail.
    # Loading checkpoint shards: 100%|██████████| 7/7 [00:01<00:00,  3.68it/s]
    # model (<class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>) has 6243584000 parameters, 6243584000 (100.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    # model (<class 'peft.peft_model.PeftModelForCausalLM'>) has 6245533696 parameters, 1949696 (0.03%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    # trainable params: 1,949,696 || all params: 6,245,533,696 || trainable%: 0.031217444255383614
    # --> model has 1.949696M params
    #
    # train_dataset: Dataset({
    #     features: ['input_ids', 'labels'],
    #     num_rows: 114599
    # })
    # val_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 1070
    # })
    # test_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 1070
    # })
    # You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it).Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model.
    # max_steps is given, it will override any value given in num_train_epochs
    # total  gpu memory:  12.72 G
    # torch  gpu memory:  11.67 G
    # tensor gpu memory:  11.66 G
    # ***** Running training *****
    #   Num examples = 114,599
    #   Num Epochs = 1
    #   Instantaneous batch size per device = 12
    #   Total train batch size (w. parallel, distributed & accumulation) = 12
    #   Gradient Accumulation steps = 1
    #   Total optimization steps = 250
    #   Number of trainable parameters = 1,949,696
    #   0%|          | 0/250 [00:00<?, ?it/s]D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    # C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:231: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
    #   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
    #   4%|▍         | 10/250 [01:39<08:27,  2.11s/it]{'loss': 4.8152, 'grad_norm': 1.8391568660736084, 'learning_rate': 4.8e-05, 'epoch': 0.0}
    #   8%|▊         | 20/250 [01:48<03:34,  1.07it/s]{'loss': 4.6387, 'grad_norm': 2.3202433586120605, 'learning_rate': 4.600000000000001e-05, 'epoch': 0.0}
    #  12%|█▏        | 30/250 [01:58<03:33,  1.03it/s]{'loss': 4.3238, 'grad_norm': 2.4035308361053467, 'learning_rate': 4.4000000000000006e-05, 'epoch': 0.0}
    #  16%|█▌        | 40/250 [02:08<03:49,  1.09s/it]{'loss': 4.192, 'grad_norm': 2.102768659591675, 'learning_rate': 4.2e-05, 'epoch': 0.0}
    #  20%|██        | 50/250 [02:18<03:21,  1.01s/it]{'loss': 4.009, 'grad_norm': 1.91422700881958, 'learning_rate': 4e-05, 'epoch': 0.01}
    #  24%|██▍       | 60/250 [02:28<03:13,  1.02s/it]{'loss': 3.8783, 'grad_norm': 1.9220069646835327, 'learning_rate': 3.8e-05, 'epoch': 0.01}
    #  28%|██▊       | 70/250 [02:38<02:50,  1.05it/s]{'loss': 3.7914, 'grad_norm': 1.9412838220596313, 'learning_rate': 3.6e-05, 'epoch': 0.01}
    #  32%|███▏      | 80/250 [02:48<03:04,  1.09s/it]{'loss': 3.7895, 'grad_norm': 1.8838335275650024, 'learning_rate': 3.4000000000000007e-05, 'epoch': 0.01}
    #  36%|███▌      | 90/250 [02:57<02:26,  1.09it/s]{'loss': 3.7295, 'grad_norm': 2.1114237308502197, 'learning_rate': 3.2000000000000005e-05, 'epoch': 0.01}
    #  40%|████      | 100/250 [03:08<02:45,  1.10s/it]{'loss': 3.7629, 'grad_norm': 2.109829902648926, 'learning_rate': 3e-05, 'epoch': 0.01}
    #  44%|████▍     | 110/250 [03:18<02:28,  1.06s/it]{'loss': 3.6922, 'grad_norm': 2.3670461177825928, 'learning_rate': 2.8000000000000003e-05, 'epoch': 0.01}
    #  48%|████▊     | 120/250 [03:28<02:10,  1.00s/it]{'loss': 3.7369, 'grad_norm': 2.396379232406616, 'learning_rate': 2.6000000000000002e-05, 'epoch': 0.01}
    #  50%|█████     | 125/250 [03:33<02:07,  1.02s/it]***** Running Evaluation *****
    #   Num examples = 50
    #   Batch size = 12
    #
    #   0%|          | 0/5 [00:00<?, ?it/s]
    #  40%|████      | 2/5 [00:06<00:10,  3.35s/it]
    #  60%|██████    | 3/5 [00:13<00:09,  4.77s/it]
    #  80%|████████  | 4/5 [00:20<00:05,  5.49s/it]
    # 100%|██████████| 5/5 [00:27<00:00,  6.08s/it]Building prefix dict from the default dictionary ...
    # Loading model from cache C:\Users\admin\AppData\Local\Temp\jieba.cache
    # Loading model cost 0.314 seconds.
    # Prefix dict has been built successfully.
    # {'eval_rouge-1': 28.395013999999996, 'eval_rouge-2': 5.604386, 'eval_rouge-l': 22.320870000000003, 'eval_bleu-4': 0.024923493122661018, 'eval_runtime': 122.2012, 'eval_samples_per_second': 0.409, 'eval_steps_per_second': 0.041, 'epoch': 0.01}
    #
    #  50%|█████     | 125/250 [05:36<02:07,  1.02s/it]
    # 100%|██████████| 5/5 [00:27<00:00,  6.08s/it]
    #                                              Checkpoint destination directory ./output\checkpoint-125 already exists and is non-empty. Saving will proceed but saved results may be invalid.
    # Saving model checkpoint to ./output\checkpoint-125
    # tokenizer config file saved in ./output\checkpoint-125\tokenizer_config.json
    # Special tokens file saved in ./output\checkpoint-125\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  52%|█████▏    | 130/250 [05:41<19:52,  9.94s/it]{'loss': 3.6742, 'grad_norm': 2.389089345932007, 'learning_rate': 2.4e-05, 'epoch': 0.01}
    #  56%|█████▌    | 140/250 [05:51<02:12,  1.21s/it]{'loss': 3.7498, 'grad_norm': 2.5621700286865234, 'learning_rate': 2.2000000000000003e-05, 'epoch': 0.01}
    #  60%|██████    | 150/250 [06:01<01:45,  1.06s/it]{'loss': 3.7176, 'grad_norm': 2.5832748413085938, 'learning_rate': 2e-05, 'epoch': 0.02}
    #  64%|██████▍   | 160/250 [06:11<01:26,  1.04it/s]{'loss': 3.7068, 'grad_norm': 2.884803295135498, 'learning_rate': 1.8e-05, 'epoch': 0.02}
    #  68%|██████▊   | 170/250 [06:22<01:20,  1.00s/it]{'loss': 3.692, 'grad_norm': 2.637817621231079, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.02}
    #  72%|███████▏  | 180/250 [06:31<01:06,  1.05it/s]{'loss': 3.7549, 'grad_norm': 2.8193984031677246, 'learning_rate': 1.4000000000000001e-05, 'epoch': 0.02}
    #  76%|███████▌  | 190/250 [06:42<01:02,  1.04s/it]{'loss': 3.8154, 'grad_norm': 2.5782933235168457, 'learning_rate': 1.2e-05, 'epoch': 0.02}
    #  80%|████████  | 200/250 [06:51<00:48,  1.04it/s]{'loss': 3.6676, 'grad_norm': 2.9160537719726562, 'learning_rate': 1e-05, 'epoch': 0.02}
    #  84%|████████▍ | 210/250 [07:00<00:35,  1.12it/s]{'loss': 3.71, 'grad_norm': 2.8538198471069336, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.02}
    #  88%|████████▊ | 220/250 [07:11<00:30,  1.01s/it]{'loss': 3.7516, 'grad_norm': 2.7598519325256348, 'learning_rate': 6e-06, 'epoch': 0.02}
    #  92%|█████████▏| 230/250 [07:21<00:21,  1.06s/it]{'loss': 3.6957, 'grad_norm': 2.9201884269714355, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.02}
    #  96%|█████████▌| 240/250 [07:30<00:09,  1.03it/s]{'loss': 3.7213, 'grad_norm': 3.0240941047668457, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.03}
    # 100%|██████████| 250/250 [07:40<00:00,  1.08it/s]***** Running Evaluation *****
    # {'loss': 3.6836, 'grad_norm': 2.846071481704712, 'learning_rate': 0.0, 'epoch': 0.03}
    #   Num examples = 50
    #   Batch size = 12
    #
    #   0%|          | 0/5 [00:00<?, ?it/s]
    #  40%|████      | 2/5 [00:06<00:10,  3.34s/it]
    #  60%|██████    | 3/5 [00:13<00:09,  4.75s/it]
    #  80%|████████  | 4/5 [00:20<00:05,  5.49s/it]
    #
    # 100%|██████████| 250/250 [09:37<00:00,  1.08it/s]
    # 100%|██████████| 5/5 [00:27<00:00,  6.09s/it]
    #                                              Checkpoint destination directory ./output\checkpoint-250 already exists and is non-empty. Saving will proceed but saved results may be invalid.
    # Saving model checkpoint to ./output\checkpoint-250
    # {'eval_rouge-1': 28.395645999999996, 'eval_rouge-2': 6.052718, 'eval_rouge-l': 22.884051999999993, 'eval_bleu-4': 0.03046700715847921, 'eval_runtime': 116.9475, 'eval_samples_per_second': 0.428, 'eval_steps_per_second': 0.043, 'epoch': 0.03}
    # tokenizer config file saved in ./output\checkpoint-250\tokenizer_config.json
    # Special tokens file saved in ./output\checkpoint-250\special_tokens_map.json
    #
    #
    # Training completed. Do not forget to share your model on huggingface.co/models =)
    #
    #
    # 100%|██████████| 250/250 [09:38<00:00,  2.31s/it]
    # {'train_runtime': 578.2234, 'train_samples_per_second': 5.188, 'train_steps_per_second': 0.432, 'train_loss': 3.8679921875, 'epoch': 0.03}
    # 'main' spent 584.3999s.
    if _with_fine_tune:
        fine_tune(data_dir=_save_dir, model_dir=_model_dir, config_file=_config_file)

    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    # 1. 尝试放松身心，如深呼吸、冥想或热水泡澡。
    # 2. 避免刺激性食物和饮料，如咖啡、茶和巧克力。
    # 3. 保持规律的睡眠时间表。
    # 4. 减少使用电子产品，特别是在睡前。
    # 5. 尝试进行轻柔的伸展运动，如瑜伽或拉伸运动。
    # 6. 如果需要，可以使用助眠药物。但在使用前，请咨询医生。
    # 7. 保持良好的睡眠环境，如安静、舒适、黑暗和凉爽。
    # 这是一款非常性感的百褶网纱连衣裙，裙下摆采用了压褶设计，能够很好地修饰身材，让穿着者显得更加苗条高挑。裙衣门襟采用了拉链设计，既方便穿脱，又增加了时尚感。整体风格为性感，适合各种场合穿着，让人眼前一亮。
    # 抱歉，我无法提供关于 "aw.fly_to()" 的具体含义，因为我无法确定这个函数或方法来自哪个编程语言或具体的项目。如果您能提供更多上下文信息，我将尽力帮助您。
    # Sure, I'll be happy to help you with the AirSim simulator for drones. Please let me know what task you would like me to complete, and I'll provide you with the necessary Python code and explanation.
    # 明白了，您是让我使用 AirSim 模拟器编写 Python 代码来控制无人机。您提供了一些函数，如 takeoff()、land()、get_drone_position()、fly_to()、fly_path()、set_yaw()、get_yaw()、get_position()，以及一些关于如何使用这些函数的说明。您还提供了一些场景和示例，以及一些关于如何处理不同类型物体的信息。最后，您让我注意不要在没有明确指令的情况下 assumptions（假设），并且要根据实际物体的位置和方向编写代码。
    # To lift the drone up by 7 meters, we can use the "aw.takeoff()" function to take off, and then use the "aw.fly_to()" function to fly to a height of 7 meters.
    #
    # Here is the Python code to accomplish this task:
    # ```python
    # # Take off the drone
    # aw.takeoff()
    #
    # # Fly to a height of 7 meters
    # aw.fly_to([0, 7, 0])
    # ```
    # The first line takes off the drone, and the second line flies the drone to a height of 7 meters. Note that we are using the "fly\_to" function to move the drone to a specific position, rather than using the "moveToPositionAsync()" or "moveToZAsync()" functions. This is because we already know the height we want to fly the drone to, and we want to use that height as the y-coordinate of the "fly\_to" function.
    print("_" * 20 + "before_fine_tune" + "_" * 20)
    before_fine_tune(_temperature=_temperature)

    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    # 1. 尝试放松身心，如冥想、深呼吸等。
    # 2. 避免咖啡因和尼古丁。
    # 3. 保持规律作息，尽量每天按时上床睡觉。
    # 4. 减少使用电子产品，尤其是睡前。
    # 5. 增加运动量，但避免剧烈运动。
    # 6. 睡前适当饮水，但避免过多。
    # 7. 尝试阅读或听轻音乐。
    # 8. 如有需要，可咨询医生或专业人士。
    # 这款连衣裙的版型设计，整体上采用套头的设计，方便穿脱，而裙身的设计，采用百褶的款式，整体上采用网纱的材质，整体上采用拉链的设计，方便穿脱，而裙下摆的设计，整体上采用压褶的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上采用收腰的设计，整体上
    # 抱歉，我无法提供关于 "aw.fly_to()" 的详细信息，因为我无法确定您所提到的 "aw" 代表什么。如果您能提供更多上下文信息，我将尽力帮助您。
    # To set the position of the drone in the AirSim environment, you can use the `set_position()` function. This function takes the x, y, and z coordinates as arguments and sets the position of the drone to those coordinates.
    #
    # Here is an example of how you can use the `set_position()` function to set the position of the drone:
    # ```
    # # Set the position of the drone to (x, y, z)
    # set_position(x=10, y=20, z=30)
    # ```
    # In this example, the drone will be set to the coordinates (10, 20, 30).
    #
    # Explanation:
    # - The `set_position()` function sets the position of the drone.
    # - The x, y, and z coordinates are passed as arguments to the function.
    # - The function updates the position of the drone to the specified coordinates.
    # 明白了，我会遵循您的指示，在使用 AirSim 进行无人机操作时遵循这些规则。在后续的对话中，我会根据您的要求使用相应的函数来操作无人机。如果您有任何问题或需要进一步的解释，请随时告诉我。
    # To raise the drone up by 7 meters, you can use the `set_position()` function and specify the z-coordinate as 7. Here's an example of how you can do this:
    # ```
    # # Raise the drone up by 7 meters
    # aw.set_position(x=0, y=0, z=7)
    # ```
    # In this example, the drone will be raised up to a height of 7 meters.
    #
    # Explanation:
    # - The `set_position()` function sets the position of the drone.
    # - The x, y, and z coordinates are passed as arguments to the function.
    # - The function updates the position of the drone to the specified coordinates.
    #
    # Please note that this will raise the drone to a new position based on the current position and the specified height. If the drone is already at a height of 7 meters or higher, this function will simply set the position of the drone to the same position.
    print("_" * 20 + "after_fine_tune" + "_" * 20)
    after_fine_tune(_checkpoint, _temperature=_temperature)


# 微调 craft_robotics 项目
def fine_tune_using_craft_robotics(_save_dir, _model_dir=CHATGLM3_6B_model_dir, _temperature=0.1,
                                   _config_file="./finetune_configs/lora_cr.yaml", _with_fine_tune=False,
                                   _checkpoint=r"./output_cr/checkpoint-100"):
    # tensorflow sed random seed fail.
    # Loading checkpoint shards: 100%|██████████| 7/7 [00:01<00:00,  3.67it/s]
    # model (<class 'transformers_modules.chatglm3-6b.modeling_chatglm.ChatGLMForConditionalGeneration'>) has 6243584000 parameters, 6243584000 (100.00%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    # model (<class 'peft.peft_model.PeftModelForCausalLM'>) has 6245533696 parameters, 1949696 (0.03%) are trainable, the dtype is torch.float16，占 11.63G 显存.
    # trainable params: 1,949,696 || all params: 6,245,533,696 || trainable%: 0.031217444255383614
    # --> model has 1.949696M params
    #
    # Generating train split: 104 examples [00:00, 6521.46 examples/s]
    # Generating validation split: 26 examples [00:00, 6522.24 examples/s]
    # Generating test split: 26 examples [00:00, 6521.46 examples/s]
    # Map: 100%|██████████| 104/104 [00:00<00:00, 5797.09 examples/s]
    # Map: 100%|██████████| 26/26 [00:00<00:00, 6521.46 examples/s]
    # Map: 100%|██████████| 26/26 [00:00<00:00, 5218.04 examples/s]
    # train_dataset: Dataset({
    #     features: ['input_ids', 'labels'],
    #     num_rows: 104
    # })
    # val_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 26
    # })
    # test_dataset: Dataset({
    #     features: ['input_ids', 'output_ids'],
    #     num_rows: 26
    # })
    # You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it).Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model.
    # max_steps is given, it will override any value given in num_train_epochs
    # total  gpu memory:  13.33 G
    # torch  gpu memory:  11.67 G
    # tensor gpu memory:  11.66 G
    # ***** Running training *****
    #   Num examples = 104
    #   Num Epochs = 12
    #   Instantaneous batch size per device = 12
    #   Total train batch size (w. parallel, distributed & accumulation) = 12
    #   Gradient Accumulation steps = 1
    #   Total optimization steps = 100
    #   Number of trainable parameters = 1,949,696
    #   0%|          | 0/100 [00:00<?, ?it/s]D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    # C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:231: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
    #   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
    #  10%|█         | 10/100 [03:18<46:59, 31.33s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-10
    # {'loss': 1.9589, 'grad_norm': 2.233711004257202, 'learning_rate': 4.5e-05, 'epoch': 1.11}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-10\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-10\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  20%|██        | 20/100 [05:00<28:07, 21.09s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-20
    # {'loss': 1.726, 'grad_norm': 2.2656426429748535, 'learning_rate': 4e-05, 'epoch': 2.22}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-20\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-20\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  30%|███       | 30/100 [06:42<17:29, 14.99s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-30
    # {'loss': 1.4106, 'grad_norm': 2.0179901123046875, 'learning_rate': 3.5e-05, 'epoch': 3.33}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-30\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-30\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  40%|████      | 40/100 [08:23<10:44, 10.74s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-40
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-40\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-40\special_tokens_map.json
    # {'loss': 1.1532, 'grad_norm': 2.0942482948303223, 'learning_rate': 3e-05, 'epoch': 4.44}
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  50%|█████     | 50/100 [10:06<06:40,  8.01s/it]***** Running Evaluation *****
    #   Num examples = 26
    #   Batch size = 12
    # {'loss': 0.9213, 'grad_norm': 1.497626543045044, 'learning_rate': 2.5e-05, 'epoch': 5.56}
    #
    #   0%|          | 0/3 [00:00<?, ?it/s]
    #  67%|██████▋   | 2/3 [00:06<00:03,  3.21s/it]
    # 100%|██████████| 3/3 [00:12<00:00,  4.25s/it]Building prefix dict from the default dictionary ...
    # Loading model from cache C:\Users\admin\AppData\Local\Temp\jieba.cache
    # Loading model cost 0.314 seconds.
    # Prefix dict has been built successfully.
    #
    #  50%|█████     | 50/100 [12:06<06:40,  8.01s/it]
    # 100%|██████████| 3/3 [00:12<00:00,  4.25s/it]
    #                                              Saving model checkpoint to ./output_cr\tmp-checkpoint-50
    # {'eval_rouge-1': 5.58535, 'eval_rouge-2': 0.2382576923076923, 'eval_rouge-l': 3.7411499999999993, 'eval_bleu-4': 0.006030430359978863, 'eval_runtime': 120.1245, 'eval_samples_per_second': 0.216, 'eval_steps_per_second': 0.025, 'epoch': 5.56}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-50\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-50\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  60%|██████    | 60/100 [13:46<04:44,  7.12s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-60
    # {'loss': 0.8227, 'grad_norm': 1.8712657690048218, 'learning_rate': 2e-05, 'epoch': 6.67}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-60\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-60\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  70%|███████   | 70/100 [15:27<02:07,  4.26s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-70
    # {'loss': 0.6921, 'grad_norm': 1.431992769241333, 'learning_rate': 1.5e-05, 'epoch': 7.78}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-70\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-70\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  80%|████████  | 80/100 [17:07<01:04,  3.24s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-80
    # {'loss': 0.6496, 'grad_norm': 1.7917826175689697, 'learning_rate': 1e-05, 'epoch': 8.89}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-80\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-80\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  90%|█████████ | 90/100 [18:48<00:27,  2.80s/it]Saving model checkpoint to ./output_cr\tmp-checkpoint-90
    # {'loss': 0.5844, 'grad_norm': 1.6837605237960815, 'learning_rate': 5e-06, 'epoch': 10.0}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-90\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-90\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    # 100%|██████████| 100/100 [21:59<00:00, 29.30s/it]***** Running Evaluation *****
    # {'loss': 0.5638, 'grad_norm': 1.6716049909591675, 'learning_rate': 0.0, 'epoch': 11.11}
    #   Num examples = 26
    #   Batch size = 12
    #
    #   0%|          | 0/3 [00:00<?, ?it/s]
    #  67%|██████▋   | 2/3 [00:07<00:03,  3.53s/it]
    #
    # 100%|██████████| 100/100 [23:54<00:00, 29.30s/it]
    # 100%|██████████| 3/3 [00:14<00:00,  5.32s/it]
    #                                              Saving model checkpoint to ./output_cr\tmp-checkpoint-100
    # {'eval_rouge-1': 8.649707692307693, 'eval_rouge-2': 0.5124961538461539, 'eval_rouge-l': 5.041776923076924, 'eval_bleu-4': 0.008850109955027092, 'eval_runtime': 115.1359, 'eval_samples_per_second': 0.226, 'eval_steps_per_second': 0.026, 'epoch': 11.11}
    # tokenizer config file saved in ./output_cr\tmp-checkpoint-100\tokenizer_config.json
    # Special tokens file saved in ./output_cr\tmp-checkpoint-100\special_tokens_map.json
    #
    #
    # Training completed. Do not forget to share your model on huggingface.co/models =)
    #
    #
    # 100%|██████████| 100/100 [23:56<00:00, 14.36s/it]
    # {'train_runtime': 1436.1053, 'train_samples_per_second': 0.836, 'train_steps_per_second': 0.07, 'train_loss': 1.04825927734375, 'epoch': 11.11}
    # 'main' spent 1442.5476s.
    if _with_fine_tune:
        fine_tune(data_dir=_save_dir, model_dir=_model_dir, config_file=_config_file)

    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    # --------------------------------------------------------------------------------
    # 1. 尝试放松身心，如深呼吸、冥想或热水泡澡。
    # 2. 避免刺激性食物和饮料，如咖啡、茶和巧克力。
    # 3. 保持规律的睡眠时间表。
    # 4. 减少使用电子设备的时间，尤其是在睡前。
    # 5. 尝试进行轻度的身体运动，如瑜伽或散步。
    # 6. 如果需要，可以使用柔和的音乐或白噪音帮助入睡。
    # 7. 避免在晚上过度兴奋，如看惊悚电影或电视节目。
    # 8. 睡前限制饮水量，以减少夜间尿频的影响。
    # 9. 建立一个舒适的睡眠环境，如保持房间凉爽、安静和黑暗。
    # 10. 如果你经常睡不着，可以考虑咨询专业医生或睡眠专家。
    # --------------------------------------------------------------------------------
    # 抱歉，作为人工智能助手，我无法确定 fly_to() 具体代表什么含义，因为我无法访问互联网进行实时查询。如果您能提供更多上下文信息，我会尽力帮助您解答。
    # --------------------------------------------------------------------------------
    # Sure, I'll be happy to help you with the AirSim simulator for drones. Please let me know what task you would like me to complete, and I'll provide you with the necessary Python code and explanation.
    # --------------------------------------------------------------------------------
    # I understand the instructions. Please let me know what task you would like me to complete.
    # --------------------------------------------------------------------------------
    # To lift the drone up by 7 meters, we can use the `aw.takeoff()` function to take off, and then use the `aw.fly_to()` function to fly to a new position. Here's the code:
    # ```python
    # aw.takeoff()
    # aw.fly_to([0, 7, 0])
    # ```
    # The `aw.takeoff()` function takes off the drone, and the `aw.fly_to()` function takes the drone to the new position specified by the `[0, 7, 0]` tuple, which is 7 meters above the current position of the drone.
    #
    # Note that we are using the `[0, 7, 0]` tuple because the yaw angle is not specified. By default, the yaw angle is set to 0 degrees. Therefore, the drone will lift off vertically and fly directly upwards. If we want the drone to lift off at an angle, we can use the `aw.set_yaw()` function to set the yaw angle before taking off. For example, if we want the drone to lift off vertically at a 45-degree angle, we can use the following code:
    # ```python
    # aw.set_yaw(45)
    # aw.takeoff()
    # aw.fly_to([0, 7, 0])
    # ```
    # This will take off the drone vertically at a 45-degree angle, and then fly directly upwards to a new position at a height of 7 meters.
    print("_" * 20 + "before_fine_tune" + "_" * 20)
    before_fine_tune(_temperature=_temperature)

    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
    # --------------------------------------------------------------------------------
    # 1. 尝试放松身心，如冥想、深呼吸等。
    # 2. 保持规律作息，尽量每天按时上床睡觉。
    # 3. 减少使用电子产品，尤其是睡前。
    # 4. 适量锻炼，但避免剧烈运动。
    # 5. 睡前适当饮水，避免过多液体摄入。
    # 6. 调整环境，保持安静、舒适。
    # 7. 避免刺激性食物、饮料。
    # 8. 尝试阅读、听轻音乐等有助于入睡。
    # 9. 如果长期失眠，请咨询专业医生。
    # --------------------------------------------------------------------------------
    # 抱歉，我无法回答您的问题，因为我不知道"fly_to()"是什么含义。如果您可以提供更多的上下文信息，我将尽力帮助您。
    # --------------------------------------------------------------------------------
    # Sure, I'll be happy to help you with the AirSim simulator for drones. Please let me know what task you would like me to complete, and I'll provide you with the necessary Python code and explanation.
    # --------------------------------------------------------------------------------
    # I understand the instructions. Please let me know what task you would like me to complete.
    # --------------------------------------------------------------------------------
    # To raise the drone up by 7 meters, we can use the `aw.fly_to()` function to move the drone to a new position above the initial position. Here's the code to achieve this:
    # ```python
    # aw.takeoff()
    # aw.fly_to([0, 0, 7])
    # ```
    # This code will take off the drone, fly it to a position 7 meters above the ground, and then land it. The `fly_to()` function will automatically calculate the path to the new position based on the drone's current position and orientation.
    #
    # Note that this code assumes that the initial position of the drone is at sea level. If the initial position is above sea level, we can simply use the `aw.fly_to()` function without taking off the drone first.
    print("_" * 20 + "after_fine_tune" + "_" * 20)
    after_fine_tune(_checkpoint, _temperature=_temperature)


@func_timer(arg=True)
def main():
    fix_all_seed(_simple=False, _warn_only=False)
    temperature = 0.01

    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\train.json 共有 114599 行.
    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\dev.json 共有 1070 行.
    # convert_adgen(BIGDATA_DATA_PATH + 'AdvertiseGen', BIGDATA_DATA_PATH + 'AdvertiseGen_fix')

    # save_dir = BIGDATA_DATA_PATH + 'AdvertiseGen_fix'
    # fine_tune_using_advertise(save_dir, _model_dir=CHATGLM3_6B_model_dir, _temperature=temperature,
    #                           _config_file="./finetune_configs/lora.yaml", _with_fine_tune=False,
    #                           _checkpoint=r"./output/checkpoint-250")

    save_dir = r'D:/PycharmProjects/xiebo/diantou/PromptCraft-Robotics/prompt/'
    fine_tune_using_craft_robotics(save_dir, _model_dir=CHATGLM3_6B_model_dir, _temperature=temperature,
                                   _config_file="./finetune_configs/lora_cr.yaml", _with_fine_tune=False,
                                   _checkpoint=r"./output_cr/checkpoint-100")


if __name__ == '__main__':
    main()
