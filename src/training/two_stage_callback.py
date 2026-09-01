from transformers import TrainerCallback


class TwoStageCallback(TrainerCallback):
    """Freeze-then-finetune training.

    Stage 1 (first `freeze_first_fraction` of total steps):
        the encoder is completely frozen, only the classification head
        is trained. This lets the randomly initialised head learn stable
        features without the encoder shifting underneath it.

    Stage 2 (remaining steps):
        every parameter is unfrozen and the whole model is fine-tuned
        with the normal (or discriminative) learning rate.

    The switch happens at a fixed step so training is a single continuous
    `trainer.train()` call (no manual two runs needed).
    """

    def __init__(
        self,
        freeze_first_fraction=0.3,
        unfreeze_at_step=None,
    ):

        # Fraction of total steps spent frozen. If 0 or None, no freezing.
        self.freeze_first_fraction = freeze_first_fraction

        # Optionally switch by absolute step instead of a fraction.
        self.unfreeze_at_step = unfreeze_at_step

        self._unfrozen = False

    def on_train_init(self, args, state, control, model=None, **kwargs):

        if not self._should_freeze():

            self._unfrozen = True

            return

        self._set_freeze(model, freeze=True)

    def on_step_begin(self, args, state, control, model=None, **kwargs):

        if self._unfrozen:

            return

        step = state.global_step

        if self._should_unfreeze(step, state.max_steps):

            self._set_freeze(model, freeze=False)

            self._unfrozen = True

            print(
                f"\n[TwoStage] Unfreezing full model at step {step} "
                f"(stage 2: fine-tune)\n"
            )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _should_freeze(self):

        frac = self.freeze_first_fraction

        return frac is not None and frac > 0

    def _should_unfreeze(self, global_step, total_steps):

        if self.unfreeze_at_step is not None:

            return global_step >= self.unfreeze_at_step

        if not total_steps:

            return True

        frac = self.freeze_first_fraction or 0.0

        return global_step >= int(total_steps * frac)

    def _set_freeze(self, model, freeze):

        for name, param in model.named_parameters():

            # Head parameters always stay trainable (both stages).
            if any(k in name for k in ("head.", "classifier.", "score.")):

                param.requires_grad = True

                continue

            param.requires_grad = not freeze

        if freeze:

            print(
                "\n[TwoStage] Stage 1: encoder frozen, " "training head only\n"
            )
