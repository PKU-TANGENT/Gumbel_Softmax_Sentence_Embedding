import sys
import math
import copy
# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import time
import senteval
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers import Trainer
from transformers.debug_utils import DebugOption
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.integrations import is_fairscale_available
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    )
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    ShardedDDPOption,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
if is_fairscale_available():
    from fairscale.optim import OSS
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
logger = logging.get_logger(__name__)

class STSTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            tmp_criterion = lambda x: "simple_head" in x or "swam" in x
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and not tmp_criterion(n)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and tmp_criterion(n)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.model.model_args.model_head_lr
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and not tmp_criterion(n)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and tmp_criterion(n)],
                    "weight_decay": 0.0,
                    "lr": self.model.model_args.model_head_lr
                },

            ]

            optimizer_cls, optimizer_kwargs = STSTrainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr", None)
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
        predict: bool = False
    ):
        self._memory_tracker.start()
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            eval_senteval_transfer=eval_senteval_transfer,
            predict=predict,
        )

        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=None,
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics



        self.log(metrics)
        return metrics

    def evaluation_loop(
        self,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
        predict: bool = False
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        # prediction_ignore_loss = self.model.model_args.prediction_ignore_loss
        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = model(**batch, inference=True, output_hidden_states=True)
                sentence_embeddings = outputs.hidden_states
            return sentence_embeddings.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        sts_tasks = ['STSBenchmark', 'SICKRelatedness','STS12', 'STS13', 'STS14', 'STS15', 'STS16']
        tasks = copy.deepcopy(sts_tasks) 
        transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        if eval_senteval_transfer or model.model_args.eval_transfer:
            tasks += transfer_tasks
        model.eval()
        results = se.eval(tasks)
        all_results = {}
        for i in tasks:
            if i in sts_tasks:
                all_results[i.lower()+"_spearman"] = results[i]["all"]["spearman"]["all"]
            elif i in transfer_tasks:
                all_results[i.lower()+"_acc"] = results[i]["acc"]
        all_results["avg_sts"] = sum(all_results[i.lower()+"_spearman"] for i in sts_tasks) / len(sts_tasks)
        if eval_senteval_transfer or model.model_args.eval_transfer:
            all_results["avg_transfer"] = sum(all_results[i.lower()+"_acc"] for i in transfer_tasks) / len(transfer_tasks)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(all_results)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=None)
