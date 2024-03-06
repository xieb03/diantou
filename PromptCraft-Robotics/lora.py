# import sys
# sys.path.append("../")
# from project_utils import *

import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import jieba
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

from util_path import *

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
app = typer.Typer(pretty_exceptions_show_locals=False)


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
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=Seq2SeqTrainingArguments(output_dir='./output')
    )
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

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
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


# TODO: Not sure if this is necessary, can set it to half
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            # if train with cpu, cast all params to fp32 instead of trainable ones.
            param.data = param.data.to(torch.float32)


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
        elif peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            model = get_peft_model(model, peft_config)
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
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

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

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)


@func_timer(arg=True)
def main():
    save_dir = BIGDATA_DATA_PATH + 'AdvertiseGen_fix'

    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\train.json 共有 114599 行.
    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\dev.json 共有 1070 行.
    # convert_adgen(BIGDATA_DATA_PATH + 'AdvertiseGen', save_dir)

    # Loading checkpoint shards: 100%|██████████| 7/7 [00:02<00:00,  2.42it/s]
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
    # ***** Running training *****
    #   Num examples = 114,599
    #   Num Epochs = 1
    #   Instantaneous batch size per device = 16
    #   Total train batch size (w. parallel, distributed & accumulation) = 16
    #   Gradient Accumulation steps = 1
    #   Total optimization steps = 200
    #   Number of trainable parameters = 1,949,696
    #   0%|          | 0/200 [00:00<?, ?it/s]D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    # C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:231: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
    #   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
    #   5%|▌         | 10/200 [01:19<05:58,  1.89s/it]{'loss': 4.8074, 'grad_norm': 1.742727279663086, 'learning_rate': 4.75e-05, 'epoch': 0.0}
    #  10%|█         | 20/200 [01:30<03:28,  1.16s/it]{'loss': 4.6371, 'grad_norm': 2.0353524684906006, 'learning_rate': 4.5e-05, 'epoch': 0.0}
    #  15%|█▌        | 30/200 [01:43<05:12,  1.84s/it]{'loss': 4.3875, 'grad_norm': 2.037071704864502, 'learning_rate': 4.25e-05, 'epoch': 0.0}
    #  20%|██        | 40/200 [02:00<04:27,  1.67s/it]{'loss': 4.1852, 'grad_norm': 1.8006170988082886, 'learning_rate': 4e-05, 'epoch': 0.01}
    #  25%|██▌       | 50/200 [02:16<03:47,  1.52s/it]{'loss': 3.965, 'grad_norm': 1.8800947666168213, 'learning_rate': 3.7500000000000003e-05, 'epoch': 0.01}
    #  30%|███       | 60/200 [02:29<03:08,  1.35s/it]{'loss': 3.8852, 'grad_norm': 1.7793691158294678, 'learning_rate': 3.5e-05, 'epoch': 0.01}
    #  35%|███▌      | 70/200 [02:40<02:28,  1.14s/it]{'loss': 3.81, 'grad_norm': 1.6698007583618164, 'learning_rate': 3.2500000000000004e-05, 'epoch': 0.01}
    #  40%|████      | 80/200 [02:53<02:32,  1.27s/it]{'loss': 3.7869, 'grad_norm': 1.7256942987442017, 'learning_rate': 3e-05, 'epoch': 0.01}
    #  45%|████▌     | 90/200 [03:06<02:49,  1.54s/it]{'loss': 3.7258, 'grad_norm': 1.9063904285430908, 'learning_rate': 2.7500000000000004e-05, 'epoch': 0.01}
    #  50%|█████     | 100/200 [03:19<02:14,  1.35s/it]***** Running Evaluation *****
    #   Num examples = 50
    #   Batch size = 16
    # {'loss': 3.6889, 'grad_norm': 2.0083062648773193, 'learning_rate': 2.5e-05, 'epoch': 0.01}
    #
    #   0%|          | 0/4 [00:00<?, ?it/s]
    #  50%|█████     | 2/4 [00:08<00:08,  4.21s/it]
    #  75%|███████▌  | 3/4 [00:19<00:06,  7.00s/it]
    # 100%|██████████| 4/4 [00:25<00:00,  6.83s/it]Building prefix dict from the default dictionary ...
    # Loading model from cache C:\Users\admin\AppData\Local\Temp\jieba.cache
    # Loading model cost 0.316 seconds.
    # Prefix dict has been built successfully.
    # {'eval_rouge-1': 29.970258000000005, 'eval_rouge-2': 5.467486, 'eval_rouge-l': 22.30071, 'eval_bleu-4': 0.025423957289888176, 'eval_runtime': 104.8078, 'eval_samples_per_second': 0.477, 'eval_steps_per_second': 0.038, 'epoch': 0.01}
    #
    #  50%|█████     | 100/200 [05:04<02:14,  1.35s/it]
    # 100%|██████████| 4/4 [00:26<00:00,  6.83s/it]
    #                                              Saving model checkpoint to ./output\tmp-checkpoint-100
    # D:\Users\admin\anaconda3\Lib\site-packages\peft\utils\save_and_load.py:154: UserWarning: Could not find a config file in D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b - will assume that the vocabulary was not modified.
    #   warnings.warn(
    # tokenizer config file saved in ./output\tmp-checkpoint-100\tokenizer_config.json
    # Special tokens file saved in ./output\tmp-checkpoint-100\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  55%|█████▌    | 110/200 [05:18<04:06,  2.74s/it]{'loss': 3.7857, 'grad_norm': 2.200469970703125, 'learning_rate': 2.25e-05, 'epoch': 0.02}
    #  60%|██████    | 120/200 [05:31<01:41,  1.27s/it]{'loss': 3.702, 'grad_norm': 2.1562118530273438, 'learning_rate': 2e-05, 'epoch': 0.02}
    #  65%|██████▌   | 130/200 [05:45<01:40,  1.43s/it]{'loss': 3.7434, 'grad_norm': 2.1666572093963623, 'learning_rate': 1.75e-05, 'epoch': 0.02}
    #  70%|███████   | 140/200 [05:58<01:15,  1.26s/it]{'loss': 3.8168, 'grad_norm': 2.3526597023010254, 'learning_rate': 1.5e-05, 'epoch': 0.02}
    #  75%|███████▌  | 150/200 [06:11<01:04,  1.30s/it]{'loss': 3.7166, 'grad_norm': 2.3969156742095947, 'learning_rate': 1.25e-05, 'epoch': 0.02}
    #  80%|████████  | 160/200 [06:25<00:54,  1.36s/it]{'loss': 3.7434, 'grad_norm': 2.368338108062744, 'learning_rate': 1e-05, 'epoch': 0.02}
    #  85%|████████▌ | 170/200 [06:42<01:00,  2.02s/it]{'loss': 3.7844, 'grad_norm': 2.3829667568206787, 'learning_rate': 7.5e-06, 'epoch': 0.02}
    #  90%|█████████ | 180/200 [06:58<00:26,  1.30s/it]{'loss': 3.7605, 'grad_norm': 2.5197689533233643, 'learning_rate': 5e-06, 'epoch': 0.03}
    #  95%|█████████▌| 190/200 [07:12<00:14,  1.46s/it]{'loss': 3.7598, 'grad_norm': 2.4845376014709473, 'learning_rate': 2.5e-06, 'epoch': 0.03}
    # 100%|██████████| 200/200 [07:32<00:00,  1.78s/it]***** Running Evaluation *****
    #   Num examples = 50
    #   Batch size = 16
    # {'loss': 3.8025, 'grad_norm': 2.603775978088379, 'learning_rate': 0.0, 'epoch': 0.03}
    #
    #   0%|          | 0/4 [00:00<?, ?it/s]
    #  50%|█████     | 2/4 [00:09<00:09,  4.61s/it]
    #  75%|███████▌  | 3/4 [00:21<00:07,  7.69s/it]
    #
    # {'eval_rouge-1': 29.645415999999997, 'eval_rouge-2': 5.852326000000001, 'eval_rouge-l': 23.031176000000002, 'eval_bleu-4': 0.028074359716300144, 'eval_runtime': 104.0097, 'eval_samples_per_second': 0.481, 'eval_steps_per_second': 0.038, 'epoch': 0.03}
    # 100%|██████████| 200/200 [09:16<00:00,  1.78s/it]
    # 100%|██████████| 4/4 [00:23<00:00,  5.79s/it]
    #                                              Saving model checkpoint to ./output\tmp-checkpoint-200
    # D:\Users\admin\anaconda3\Lib\site-packages\peft\utils\save_and_load.py:154: UserWarning: Could not find a config file in D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b - will assume that the vocabulary was not modified.
    #   warnings.warn(
    # tokenizer config file saved in ./output\tmp-checkpoint-200\tokenizer_config.json
    # Special tokens file saved in ./output\tmp-checkpoint-200\special_tokens_map.json
    #
    #
    # Training completed. Do not forget to share your model on huggingface.co/models =)
    #
    #
    # 100%|██████████| 200/200 [09:17<00:00,  2.79s/it]
    # ***** Running Prediction *****
    #   Num examples = 1070
    #   Batch size = 16
    # {'train_runtime': 557.2423, 'train_samples_per_second': 5.743, 'train_steps_per_second': 0.359, 'train_loss': 3.924697265625, 'epoch': 0.03}
    # 100%|██████████| 67/67 [11:11<00:00, 10.02s/it]
    # 'main' spent 1312.5112s.
    fine_tune(data_dir=save_dir, model_dir=CHATGLM3_6B_model_dir, config_file="./finetune_configs/lora.yaml")


if __name__ == '__main__':
    main()
