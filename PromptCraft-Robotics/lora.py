import dataclasses as dc
import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import jieba
import ruamel.yaml as yaml
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

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)


@func_timer(arg=True)
def main():
    fix_all_seed(_simple=False)

    save_dir = BIGDATA_DATA_PATH + 'AdvertiseGen_fix'

    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\train.json 共有 114599 行.
    # D:\PycharmProjects\xiebo\diantou\bigdata\data\AdvertiseGen\dev.json 共有 1070 行.
    # convert_adgen(BIGDATA_DATA_PATH + 'AdvertiseGen', save_dir)

    # tensorflow sed random seed fail.
    # Loading checkpoint shards: 100%|██████████| 7/7 [00:02<00:00,  3.37it/s]
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
    # total  gpu memory:  12.9 G
    # torch  gpu memory:  11.67 G
    # tensor gpu memory:  11.66 G
    # ***** Running training *****
    #   Num examples = 114,599
    #   Num Epochs = 1
    #   Instantaneous batch size per device = 8
    #   Total train batch size (w. parallel, distributed & accumulation) = 8
    #   Gradient Accumulation steps = 1
    #   Total optimization steps = 400
    #   Number of trainable parameters = 1,949,696
    #   0%|          | 0/400 [00:00<?, ?it/s]D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    # C:\Users\admin\.cache\huggingface\modules\transformers_modules\chatglm3-6b\modeling_chatglm.py:231: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at ..\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
    #   context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
    #   2%|▎         | 10/400 [01:50<15:45,  2.43s/it]{'loss': 4.7887, 'grad_norm': 2.1173408031463623, 'learning_rate': 4.875e-05, 'epoch': 0.0}
    #   5%|▌         | 20/400 [02:00<06:52,  1.08s/it]{'loss': 4.6477, 'grad_norm': 2.4829466342926025, 'learning_rate': 4.75e-05, 'epoch': 0.0}
    #   8%|▊         | 30/400 [02:11<06:10,  1.00s/it]{'loss': 4.418, 'grad_norm': 2.719491958618164, 'learning_rate': 4.6250000000000006e-05, 'epoch': 0.0}
    #  10%|█         | 40/400 [02:22<07:12,  1.20s/it]{'loss': 4.1482, 'grad_norm': 2.392939329147339, 'learning_rate': 4.5e-05, 'epoch': 0.0}
    #  12%|█▎        | 50/400 [02:33<06:12,  1.06s/it]{'loss': 3.9312, 'grad_norm': 2.4369444847106934, 'learning_rate': 4.375e-05, 'epoch': 0.0}
    #  15%|█▌        | 60/400 [02:45<07:03,  1.25s/it]{'loss': 3.9537, 'grad_norm': 2.512172222137451, 'learning_rate': 4.25e-05, 'epoch': 0.0}
    #  18%|█▊        | 70/400 [02:55<05:57,  1.08s/it]{'loss': 3.8322, 'grad_norm': 2.3900489807128906, 'learning_rate': 4.125e-05, 'epoch': 0.0}
    #  20%|██        | 80/400 [03:06<06:04,  1.14s/it]{'loss': 3.8352, 'grad_norm': 2.141808271408081, 'learning_rate': 4e-05, 'epoch': 0.01}
    #  22%|██▎       | 90/400 [03:19<06:05,  1.18s/it]{'loss': 3.7057, 'grad_norm': 2.4001009464263916, 'learning_rate': 3.875e-05, 'epoch': 0.01}
    #  25%|██▌       | 100/400 [03:30<05:47,  1.16s/it]{'loss': 3.7051, 'grad_norm': 2.375187397003174, 'learning_rate': 3.7500000000000003e-05, 'epoch': 0.01}
    #  28%|██▊       | 110/400 [03:40<05:08,  1.06s/it]{'loss': 3.7369, 'grad_norm': 2.576702356338501, 'learning_rate': 3.625e-05, 'epoch': 0.01}
    #  30%|███       | 120/400 [03:52<05:42,  1.22s/it]{'loss': 3.698, 'grad_norm': 2.6313319206237793, 'learning_rate': 3.5e-05, 'epoch': 0.01}
    #  32%|███▎      | 130/400 [04:04<05:07,  1.14s/it]{'loss': 3.6764, 'grad_norm': 2.9250454902648926, 'learning_rate': 3.375000000000001e-05, 'epoch': 0.01}
    #  35%|███▌      | 140/400 [04:14<04:45,  1.10s/it]{'loss': 3.7215, 'grad_norm': 2.929673194885254, 'learning_rate': 3.2500000000000004e-05, 'epoch': 0.01}
    #  38%|███▊      | 150/400 [04:26<05:13,  1.25s/it]{'loss': 3.7186, 'grad_norm': 2.9973206520080566, 'learning_rate': 3.125e-05, 'epoch': 0.01}
    #  40%|████      | 160/400 [04:37<04:32,  1.14s/it]{'loss': 3.6934, 'grad_norm': 3.1966705322265625, 'learning_rate': 3e-05, 'epoch': 0.01}
    #  42%|████▎     | 170/400 [04:49<04:20,  1.13s/it]{'loss': 3.598, 'grad_norm': 3.132033586502075, 'learning_rate': 2.8749999999999997e-05, 'epoch': 0.01}
    #  45%|████▌     | 180/400 [05:00<03:59,  1.09s/it]{'loss': 3.6859, 'grad_norm': 3.351003408432007, 'learning_rate': 2.7500000000000004e-05, 'epoch': 0.01}
    #  48%|████▊     | 190/400 [05:11<04:05,  1.17s/it]{'loss': 3.5826, 'grad_norm': 3.148512840270996, 'learning_rate': 2.625e-05, 'epoch': 0.01}
    #  50%|█████     | 200/400 [05:23<03:39,  1.10s/it]***** Running Evaluation *****
    #   Num examples = 50
    #   Batch size = 8
    # {'loss': 3.6367, 'grad_norm': 3.2902638912200928, 'learning_rate': 2.5e-05, 'epoch': 0.01}
    #
    #   0%|          | 0/7 [00:00<?, ?it/s]
    #  29%|██▊       | 2/7 [00:06<00:16,  3.27s/it]
    #  43%|████▎     | 3/7 [00:13<00:18,  4.65s/it]
    #  57%|█████▋    | 4/7 [00:15<00:11,  3.70s/it]
    #  71%|███████▏  | 5/7 [00:17<00:06,  3.22s/it]
    #  86%|████████▌ | 6/7 [00:24<00:04,  4.33s/it]
    # 100%|██████████| 7/7 [00:26<00:00,  3.80s/it]Building prefix dict from the default dictionary ...
    # Loading model from cache C:\Users\admin\AppData\Local\Temp\jieba.cache
    # Loading model cost 0.317 seconds.
    # Prefix dict has been built successfully.
    # {'eval_rouge-1': 29.266655999999994, 'eval_rouge-2': 5.416386, 'eval_rouge-l': 22.596558, 'eval_bleu-4': 0.02592826276710041, 'eval_runtime': 133.2774, 'eval_samples_per_second': 0.375, 'eval_steps_per_second': 0.053, 'epoch': 0.01}
    #
    #  50%|█████     | 200/400 [07:36<03:39,  1.10s/it]
    # 100%|██████████| 7/7 [00:27<00:00,  3.80s/it]
    #                                              Checkpoint destination directory ./output\checkpoint-200 already exists and is non-empty. Saving will proceed but saved results may be invalid.
    # Saving model checkpoint to ./output\checkpoint-200
    # D:\Users\admin\anaconda3\Lib\site-packages\peft\utils\save_and_load.py:154: UserWarning: Could not find a config file in D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b - will assume that the vocabulary was not modified.
    #   warnings.warn(
    # tokenizer config file saved in ./output\checkpoint-200\tokenizer_config.json
    # Special tokens file saved in ./output\checkpoint-200\special_tokens_map.json
    # D:\Users\admin\anaconda3\Lib\site-packages\torch\utils\checkpoint.py:460: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
    #   warnings.warn(
    #  52%|█████▎    | 210/400 [07:47<08:44,  2.76s/it]{'loss': 3.7338, 'grad_norm': 3.5706963539123535, 'learning_rate': 2.375e-05, 'epoch': 0.01}
    #  55%|█████▌    | 220/400 [07:58<03:25,  1.14s/it]{'loss': 3.6848, 'grad_norm': 3.5871829986572266, 'learning_rate': 2.25e-05, 'epoch': 0.02}
    #  57%|█████▊    | 230/400 [08:08<03:06,  1.09s/it]{'loss': 3.5541, 'grad_norm': 3.4223954677581787, 'learning_rate': 2.125e-05, 'epoch': 0.02}
    #  60%|██████    | 240/400 [08:19<02:55,  1.10s/it]{'loss': 3.6383, 'grad_norm': 4.052501678466797, 'learning_rate': 2e-05, 'epoch': 0.02}
    #  62%|██████▎   | 250/400 [08:31<03:07,  1.25s/it]{'loss': 3.6693, 'grad_norm': 3.7695517539978027, 'learning_rate': 1.8750000000000002e-05, 'epoch': 0.02}
    #  65%|██████▌   | 260/400 [08:42<02:28,  1.06s/it]{'loss': 3.5732, 'grad_norm': 3.6141488552093506, 'learning_rate': 1.75e-05, 'epoch': 0.02}
    #  68%|██████▊   | 270/400 [08:53<02:19,  1.07s/it]{'loss': 3.667, 'grad_norm': 3.900953769683838, 'learning_rate': 1.6250000000000002e-05, 'epoch': 0.02}
    #  70%|███████   | 280/400 [09:04<02:13,  1.11s/it]{'loss': 3.7475, 'grad_norm': 3.5460364818573, 'learning_rate': 1.5e-05, 'epoch': 0.02}
    #  72%|███████▎  | 290/400 [09:16<02:11,  1.20s/it]{'loss': 3.5783, 'grad_norm': 3.624561071395874, 'learning_rate': 1.3750000000000002e-05, 'epoch': 0.02}
    #  75%|███████▌  | 300/400 [09:26<01:46,  1.06s/it]{'loss': 3.6057, 'grad_norm': 4.056458473205566, 'learning_rate': 1.25e-05, 'epoch': 0.02}
    #  78%|███████▊  | 310/400 [09:37<01:40,  1.11s/it]{'loss': 3.6078, 'grad_norm': 3.95469069480896, 'learning_rate': 1.125e-05, 'epoch': 0.02}
    #  80%|████████  | 320/400 [09:47<01:23,  1.04s/it]{'loss': 3.5979, 'grad_norm': 3.535107374191284, 'learning_rate': 1e-05, 'epoch': 0.02}
    #  82%|████████▎ | 330/400 [09:59<01:21,  1.16s/it]{'loss': 3.6783, 'grad_norm': 3.5198938846588135, 'learning_rate': 8.75e-06, 'epoch': 0.02}
    #  85%|████████▌ | 340/400 [10:10<01:05,  1.10s/it]{'loss': 3.5963, 'grad_norm': 3.640310049057007, 'learning_rate': 7.5e-06, 'epoch': 0.02}
    #  88%|████████▊ | 350/400 [10:21<00:55,  1.12s/it]{'loss': 3.5387, 'grad_norm': 3.580289125442505, 'learning_rate': 6.25e-06, 'epoch': 0.02}
    #  90%|█████████ | 360/400 [10:32<00:43,  1.10s/it]{'loss': 3.6697, 'grad_norm': 4.112462520599365, 'learning_rate': 5e-06, 'epoch': 0.03}
    #  92%|█████████▎| 370/400 [10:42<00:32,  1.09s/it]{'loss': 3.5717, 'grad_norm': 3.465850353240967, 'learning_rate': 3.75e-06, 'epoch': 0.03}
    #  95%|█████████▌| 380/400 [10:53<00:22,  1.12s/it]{'loss': 3.6156, 'grad_norm': 3.556001901626587, 'learning_rate': 2.5e-06, 'epoch': 0.03}
    #  98%|█████████▊| 390/400 [11:05<00:11,  1.20s/it]{'loss': 3.7178, 'grad_norm': 4.060969829559326, 'learning_rate': 1.25e-06, 'epoch': 0.03}
    # 100%|██████████| 400/400 [11:17<00:00,  1.30s/it]***** Running Evaluation *****
    #   Num examples = 50
    #   Batch size = 8
    # {'loss': 3.6064, 'grad_norm': 3.7951302528381348, 'learning_rate': 0.0, 'epoch': 0.03}
    #
    #   0%|          | 0/7 [00:00<?, ?it/s]
    #  29%|██▊       | 2/7 [00:06<00:16,  3.27s/it]
    #  43%|████▎     | 3/7 [00:08<00:11,  2.90s/it]
    #  57%|█████▋    | 4/7 [00:10<00:07,  2.59s/it]
    #  71%|███████▏  | 5/7 [00:12<00:04,  2.31s/it]
    #  86%|████████▌ | 6/7 [00:19<00:03,  3.69s/it]
    #
    # {'eval_rouge-1': 28.351001999999998, 'eval_rouge-2': 5.814047999999999, 'eval_rouge-l': 22.311304, 'eval_bleu-4': 0.0281416633161164, 'eval_runtime': 128.6532, 'eval_samples_per_second': 0.389, 'eval_steps_per_second': 0.054, 'epoch': 0.03}
    # 100%|██████████| 400/400 [13:25<00:00,  1.30s/it]
    # 100%|██████████| 7/7 [00:22<00:00,  3.53s/it]
    #                                              Checkpoint destination directory ./output\checkpoint-400 already exists and is non-empty. Saving will proceed but saved results may be invalid.
    # Saving model checkpoint to ./output\checkpoint-400
    # D:\Users\admin\anaconda3\Lib\site-packages\peft\utils\save_and_load.py:154: UserWarning: Could not find a config file in D:\PycharmProjects\xiebo\diantou\bigdata\models\ZhipuAI\chatglm3-6b - will assume that the vocabulary was not modified.
    #   warnings.warn(
    # tokenizer config file saved in ./output\checkpoint-400\tokenizer_config.json
    # Special tokens file saved in ./output\checkpoint-400\special_tokens_map.json
    #
    #
    # Training completed. Do not forget to share your model on huggingface.co/models =)
    #
    #
    # 100%|██████████| 400/400 [13:26<00:00,  2.02s/it]
    # ***** Running Prediction *****
    #   Num examples = 1070
    #   Batch size = 8
    # {'train_runtime': 806.7695, 'train_samples_per_second': 3.966, 'train_steps_per_second': 0.496, 'train_loss': 3.7591455078125, 'epoch': 0.03}
    # 100%|██████████| 134/134 [10:31<00:00,  4.72s/it]
    # 'main' spent 1549.5834s.
    fine_tune(data_dir=save_dir, model_dir=CHATGLM3_6B_model_dir, config_file="./finetune_configs/lora.yaml")


if __name__ == '__main__':
    main()
