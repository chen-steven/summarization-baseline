from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, is_torch_tpu_available, EvalPrediction
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat
from transformers.trainer_utils import PredictionOutput
import logging
import collections

logger = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

class ExtractorAbstractorTrainer(Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "decoder_sentence_indicator": inputs['sentence_indicator'],
            "decoder_sentence_labels": inputs['sentence_labels'],
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        sentence_labels = inputs['sentence_labels']
        gumbel_output = outputs.gumbel_output
        sentence_indicator = inputs['sentence_indicator']
        extracted_attentions = outputs.extracted_attentions.long()
        extracted_tokens = inputs['input_ids']*extracted_attentions + (1-extracted_attentions)*self.tokenizer.pad_token_id
        
        

        return (loss, generated_tokens, labels, gumbel_output, sentence_labels, extracted_tokens)

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # Note: in torch.distributed mode, there's no point in wrapping the model
            # inside a DistributedDataParallel as we'll be under `no_grad` anyways

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        gumbel_host: Union[torch.Tensor, List[torch.Tensor]] = None
        sentence_labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        sentence_indicator_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
            gumbel_gatherer = DistributedTensorGatherer(world_size, num_examples)
            sentence_labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
            sentence_indicator_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels, gumbel_output, sentence_labels, sentence_indicator = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if gumbel_output is not None:
                gumbel_host = gumbel_output if gumbel_host is None else nested_concat(gumbel_host, gumbel_output, padding_index=-1)
            if sentence_labels is not None:
                sentence_labels_host = sentence_labels if sentence_labels_host is None else nested_concat(sentence_labels_host, sentence_labels, padding_index=-1)
            if sentence_indicator is not None:
                sentence_indicator_host = sentence_indicator if sentence_indicator_host is None else nested_concat(sentence_indicator_host, sentence_indicator, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    gumbel_gatherer.add_arrays(self._gather_and_numpify(gumbel_host, "eval_gumbel_output"))
                    sentence_labels_gatherer.add_arrays(self._gather_and_numpify(sentence_labels_host, "eval_sentence_idxs"))
                    sentence_indicator_gatherer.add_arrays(self._gather_and_numpify(sentence_indicator_host, "eval_sentence_indicator"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, gumbel_host, sentence_labels_host, sentence_indicator_host = None, None, None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            gumbel_gatherer.add_arrays(self._gather_and_numpify(gumbel_host, "eval_gumbel_output"))
            sentence_labels_gatherer.add_arrays(self._gather_and_numpify(sentence_labels_host, "eval_sentence_idxs"))
            sentence_indicator_gatherer.add_arrays(self._gather_and_numpify(sentence_indicator_host, "eval_sentence_indicator"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        gumbel_outputs = gumbel_gatherer.finalize() if not prediction_loss_only else None
        sentence_idxs = sentence_labels_gatherer.finalize() if not prediction_loss_only else None
        sentence_indicators = sentence_indicator_gatherer.finalize() if not prediction_loss_only else None
        print(sentence_idxs, 'test')

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(preds, label_ids, gumbel_outputs, sentence_idxs, sentence_indicators)
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

