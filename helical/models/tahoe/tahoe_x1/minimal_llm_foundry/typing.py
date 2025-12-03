# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Base class for Enums containing string values."""
from __future__ import annotations

import textwrap
import warnings
from enum import Enum

from abc import ABC, abstractmethod
from typing import Any
from typing import Any

class Serializable:
    """Interface for serialization; used by checkpointing."""

    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary representing the internal state.

        The returned dictionary must be pickale-able via :func:`torch.save`.

        Returns:
            dict[str, Any]: The state of the object.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the state of the object.

        Args:
            state (dict[str, Any]): The state of the object, as previously returned by :meth:`.state_dict`.
        """
        pass

class StringEnum(Enum):
    """Base class for Enums containing string values.

    This class enforces that all keys are uppercase and all values are lowercase. It also offers
    the following convenience features:

    *   ``StringEnum(value)`` will perform a case-insensitive match on both the keys and value,
        and is a no-op if given an existing instance of the class.

        .. testsetup::

            import warnings

            warnings.filterwarnings(action="ignore", message="Detected comparison between a string")

        .. doctest::

            >>> from composer.utils import StringEnum
            >>> class MyStringEnum(StringEnum):
            ...     KEY = "value"
            >>> MyStringEnum("KeY")  # case-insensitive match on the key
            <MyStringEnum.KEY: 'value'>
            >>> MyStringEnum("VaLuE")  # case-insensitive match on the value
            <MyStringEnum.KEY: 'value'>
            >>> MyStringEnum(MyStringEnum.KEY)  # no-op if given an existing instance
            <MyStringEnum.KEY: 'value'>

        .. testcleanup::

            warnings.resetwarnings()

    *   Equality checks support case-insensitive comparisons against strings:

        .. testsetup::

            import warnings

            warnings.filterwarnings(action="ignore", message="Detected comparison between a string")

        .. doctest::

            >>> from composer.utils import StringEnum
            >>> class MyStringEnum(StringEnum):
            ...     KEY = "value"
            >>> MyStringEnum.KEY == "KeY"  # case-insensitive match on the key
            True
            >>> MyStringEnum.KEY == "VaLuE"  # case-insensitive match on the value
            True
            >>> MyStringEnum.KEY == "something else"
            False

        .. testcleanup::

            warnings.resetwarnings()
    """
    __hash__ = Enum.__hash__  # pyright: ignore[reportGeneralTypeIssues]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            cls_name = self.__class__.__name__
            warnings.warn(
                f"Detected comparison between a string and {cls_name}. Please use {cls_name}('{other}') "
                f'to convert both types to {cls_name} before comparing.',
                category=UserWarning,
            )
            try:
                o_enum = type(self)(other)
            except ValueError:  # `other` is not a valid enum option
                return NotImplemented
            return super().__eq__(o_enum)
        return super().__eq__(other)

    def __init__(self, *args: object) -> None:
        if self.name.upper() != self.name:
            raise ValueError(
                textwrap.dedent(
                    f"""\
                {self.__class__.__name__}.{self.name} is invalid.
                All keys in {self.__class__.__name__} must be uppercase.
                To fix, rename to '{self.name.upper()}'.""",
                ),
            )
        if self.value.lower() != self.value:
            raise ValueError(
                textwrap.dedent(
                    f"""\
                The value for {self.__class__.__name__}.{self.name}={self.value} is invalid.
                All values in {self.__class__.__name__} must be lowercase. "
                To fix, rename to '{self.value.lower()}'.""",
                ),
            )

    @classmethod
    def _missing_(cls, value: object) -> StringEnum:
        # Override _missing_ so both lowercase and uppercase names are supported,
        # as well as passing an instance through
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                if value.lower() != value:
                    return cls(value.lower())
                raise ValueError(f'Value {value} not found in {cls.__name__}')
        raise TypeError(f'Unable to convert value({value}) of type {type(value)} into {cls.__name__}')



class Event(StringEnum):
    """Enum to represent training loop events.

    Events mark specific points in the training loop where an :class:`~.core.Algorithm` and :class:`~.core.Callback`
    can run.

    The following pseudocode shows where each event fires in the training loop:

    .. code-block:: python

        # <INIT>
        # <BEFORE_LOAD>
        # <AFTER_LOAD>
        # <FIT_START>
        for iteration in range(NUM_ITERATIONS):
            # <ITERATION_START>
            for epoch in range(NUM_EPOCHS):
                # <EPOCH_START>
                while True:
                    # <BEFORE_DATALOADER>
                    batch = next(dataloader)
                    if batch is None:
                        break
                    # <AFTER_DATALOADER>

                    # <BATCH_START>

                    # <BEFORE_TRAIN_BATCH>

                    for microbatch in batch.split(device_train_microbatch_size):

                        # <BEFORE_FORWARD>
                        outputs = model(batch)
                        # <AFTER_FORWARD>

                        # <BEFORE_LOSS>
                        loss = model.loss(outputs, batch)
                        # <AFTER_LOSS>

                        # <BEFORE_BACKWARD>
                        loss.backward()
                        # <AFTER_BACKWARD>

                    # Un-scale gradients

                    # <AFTER_TRAIN_BATCH>
                    optimizer.step()

                    # <BATCH_END>

                    # <EVAL_BEFORE_ALL>
                    for eval_dataloader in eval_dataloaders:
                        if should_eval(batch=True):
                            # <EVAL_START>
                            for batch in eval_dataloader:
                                # <EVAL_BATCH_START>
                                # <EVAL_BEFORE_FORWARD>
                                outputs, targets = model(batch)
                                # <EVAL_AFTER_FORWARD>
                                metrics.update(outputs, targets)
                                # <EVAL_BATCH_END>
                            # <EVAL_END>

                    # <EVAL_AFTER_ALL>

                    # <BATCH_CHECKPOINT>
                # <EPOCH_END>

                # <BEFORE_EVAL_ALL>
                for eval_dataloader in eval_dataloaders:
                    if should_eval(batch=True):
                        # <EVAL_START>
                        for batch in eval_dataloader:
                            # <EVAL_BATCH_START>
                            # <EVAL_BEFORE_FORWARD>
                            outputs, targets = model(batch)
                            # <EVAL_AFTER_FORWARD>
                            metrics.update(outputs, targets)
                            # <EVAL_BATCH_END>
                        # <EVAL_END>

                # <AFTER_EVAL_ALL>

                # <EPOCH_CHECKPOINT>
            # <ITERATION_END>
            # <ITERATION_CHECKPOINT>
        # <FIT_END>

    Attributes:
        INIT: Invoked in the constructor of :class:`~.trainer.Trainer`. Model surgery (see
            :mod:`~composer.utils.module_surgery`) typically occurs here.
        BEFORE_LOAD: Immediately before the checkpoint is loaded in :class:`~.trainer.Trainer`.
        AFTER_LOAD: Immediately after checkpoint is loaded in constructor of :class:`~.trainer.Trainer`.
        FIT_START: Invoked at the beginning of each call to :meth:`.Trainer.fit`. Dataset transformations typically
            occur here.
        ITERATION_START: Start of an iteration.
        EPOCH_START: Start of an epoch.
        BEFORE_DATALOADER: Immediately before the dataloader is called.
        AFTER_DATALOADER: Immediately after the dataloader is called.  Typically used for on-GPU dataloader transforms.
        BATCH_START: Start of a batch.
        BEFORE_TRAIN_BATCH: Before the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this is still called only once.
        BEFORE_FORWARD: Before the call to ``model.forward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_FORWARD: After the call to ``model.forward()``.
            This is called multiple times per batch when using gradient accumulation.
        BEFORE_LOSS: Before the call to ``model.loss()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_LOSS: After the call to ``model.loss()``.
            This is called multiple times per batch when using gradient accumulation.
        BEFORE_BACKWARD: Before the call to ``loss.backward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_BACKWARD: After the call to ``loss.backward()``.
            This is called multiple times per batch when using gradient accumulation.
        AFTER_TRAIN_BATCH: After the forward-loss-backward computation for a training batch. When using gradient
            accumulation, this event still fires only once.
        BATCH_END: End of a batch, which occurs after the optimizer step and any gradient scaling.
        BATCH_CHECKPOINT: After :attr:`.Event.BATCH_END` and any batch-wise evaluation. Saving checkpoints at this
            event allows the checkpoint saver to use the results from any batch-wise evaluation to determine whether
            a checkpoint should be saved.
        EPOCH_END: End of an epoch.
        EPOCH_CHECKPOINT: After :attr:`.Event.EPOCH_END` and any epoch-wise evaluation. Saving checkpoints at this
            event allows the checkpoint saver to use the results from any epoch-wise evaluation to determine whether
            a checkpoint should be saved.
        ITERATION_END: End of an iteration.
        ITERATION_CHECKPOINT: After :attr:`.Event.ITERATION_END`. Saving checkpoints at this event allows the checkpoint
        saver to determine whether a checkpoint should be saved.
        FIT_END: Invoked at the end of each call to :meth:`.Trainer.fit`. This event exists primarily for logging information
            and flushing callbacks. Algorithms should not transform the training state on this event, as any changes will not
            be preserved in checkpoints.

        EVAL_BEFORE_ALL: Before any evaluators process validation dataset.
        EVAL_START: Start of evaluation through the validation dataset.
        EVAL_BATCH_START: Before the call to ``model.eval_forward(batch)``
        EVAL_BEFORE_FORWARD: Before the call to ``model.eval_forward(batch)``
        EVAL_AFTER_FORWARD: After the call to ``model.eval_forward(batch)``
        EVAL_BATCH_END: After the call to ``model.eval_forward(batch)``
        EVAL_END: End of evaluation through the validation dataset.
        EVAL_AFTER_ALL: After all evaluators process validation dataset.

        EVAL_STANDALONE_START: Start of evaluation through a direct call to `trainer.eval`.
        EVAL_STANDALONE_END: End of evaluation through a direct call to `trainer.eval`.
    """

    INIT = 'init'
    BEFORE_LOAD = 'before_load'
    AFTER_LOAD = 'after_load'
    FIT_START = 'fit_start'

    ITERATION_START = 'iteration_start'

    EPOCH_START = 'epoch_start'

    BEFORE_DATALOADER = 'before_dataloader'
    AFTER_DATALOADER = 'after_dataloader'

    BATCH_START = 'batch_start'

    BEFORE_TRAIN_BATCH = 'before_train_batch'

    BEFORE_FORWARD = 'before_forward'
    AFTER_FORWARD = 'after_forward'

    BEFORE_LOSS = 'before_loss'
    AFTER_LOSS = 'after_loss'

    BEFORE_BACKWARD = 'before_backward'
    AFTER_BACKWARD = 'after_backward'

    AFTER_TRAIN_BATCH = 'after_train_batch'

    BATCH_END = 'batch_end'
    BATCH_CHECKPOINT = 'batch_checkpoint'

    EPOCH_END = 'epoch_end'
    EPOCH_CHECKPOINT = 'epoch_checkpoint'

    ITERATION_END = 'iteration_end'
    ITERATION_CHECKPOINT = 'iteration_checkpoint'

    FIT_END = 'fit_end'

    EVAL_BEFORE_ALL = 'eval_before_all'
    EVAL_START = 'eval_start'
    EVAL_BATCH_START = 'eval_batch_start'
    EVAL_BEFORE_FORWARD = 'eval_before_forward'
    EVAL_AFTER_FORWARD = 'eval_after_forward'
    EVAL_BATCH_END = 'eval_batch_end'
    EVAL_END = 'eval_end'
    EVAL_AFTER_ALL = 'eval_after_all'

    EVAL_STANDALONE_START = 'eval_standalone_start'
    EVAL_STANDALONE_END = 'eval_standalone_end'

    PREDICT_START = 'predict_start'
    PREDICT_BATCH_START = 'predict_batch_start'
    PREDICT_BEFORE_FORWARD = 'predict_before_forward'
    PREDICT_AFTER_FORWARD = 'predict_after_forward'
    PREDICT_BATCH_END = 'predict_batch_end'
    PREDICT_END = 'predict_end'

    @property
    def is_before_event(self) -> bool:
        """Whether the event is an "before" event.

        An "before" event (e.g., :attr:`~Event.BEFORE_LOSS`) has a corresponding "after" event
        (.e.g., :attr:`~Event.AFTER_LOSS`).
        """
        return self in _BEFORE_EVENTS

    @property
    def is_after_event(self) -> bool:
        """Whether the event is an "after" event.

        An "after" event (e.g., :attr:`~Event.AFTER_LOSS`) has a corresponding "before" event
        (.e.g., :attr:`~Event.BEFORE_LOSS`).
        """
        return self in _AFTER_EVENTS

    @property
    def canonical_name(self) -> str:
        """The name of the event, without before/after markers.

        Events that have a corresponding "before" or "after" event share the same canonical name.

        Example:
            >>> Event.EPOCH_START.canonical_name
            'epoch'
            >>> Event.EPOCH_END.canonical_name
            'epoch'

        Returns:
            str: The canonical name of the event.
        """
        name: str = self.value
        name = name.replace('before_', '')
        name = name.replace('after_', '')
        name = name.replace('_start', '')
        name = name.replace('_end', '')
        return name

    @property
    def is_predict(self) -> bool:
        """Whether the event is during the predict loop."""
        return self.value.startswith('predict')

    @property
    def is_eval(self) -> bool:
        """Whether the event is during the eval loop."""
        return self.value.startswith('eval')


_BEFORE_EVENTS = (
    Event.BEFORE_LOAD,
    Event.FIT_START,
    Event.ITERATION_START,
    Event.EPOCH_START,
    Event.BEFORE_DATALOADER,
    Event.BATCH_START,
    Event.BEFORE_TRAIN_BATCH,
    Event.BEFORE_FORWARD,
    Event.BEFORE_LOSS,
    Event.BEFORE_BACKWARD,
    Event.EVAL_BEFORE_ALL,
    Event.EVAL_START,
    Event.EVAL_BATCH_START,
    Event.EVAL_BEFORE_FORWARD,
    Event.PREDICT_START,
    Event.PREDICT_BATCH_START,
    Event.PREDICT_BEFORE_FORWARD,
    Event.EVAL_STANDALONE_START,
)
_AFTER_EVENTS = (
    Event.AFTER_LOAD,
    Event.ITERATION_END,
    Event.EPOCH_END,
    Event.BATCH_END,
    Event.AFTER_DATALOADER,
    Event.AFTER_TRAIN_BATCH,
    Event.AFTER_FORWARD,
    Event.AFTER_LOSS,
    Event.AFTER_BACKWARD,
    Event.EVAL_AFTER_ALL,
    Event.EVAL_END,
    Event.EVAL_BATCH_END,
    Event.EVAL_AFTER_FORWARD,
    Event.FIT_END,
    Event.PREDICT_END,
    Event.PREDICT_BATCH_END,
    Event.PREDICT_AFTER_FORWARD,
    Event.EVAL_STANDALONE_END,
)

class Precision(StringEnum):
    """Enum class for the numerical precision to be used by the model.

    Attributes:
        FP32: Use 32-bit floating-point precision. Compatible with CPUs and GPUs.
        AMP_FP16: Use :mod:`torch.amp` with 16-bit floating-point precision. Only compatible
            with GPUs.
        AMP_BF16: Use :mod:`torch.amp` with 16-bit BFloat precision.
        AMP_FP8: Use :mod:`transformer_engine.pytorch.fp8_autocast` with 8-bit FP8 precison.
    """
    FP32 = 'fp32'
    AMP_FP16 = 'amp_fp16'
    AMP_BF16 = 'amp_bf16'
    AMP_FP8 = 'amp_fp8'


class Algorithm(Serializable, ABC):
    """Base class for algorithms.

    Algorithms are pieces of code which run at specific events (see :class:`.Event`) in the training loop.
    Algorithms modify the trainer's :class:`.State`, generally with the effect of improving the model's quality
    or increasing the efficiency and throughput of the training loop.

    Algorithms must implement the following two methods:
      +----------------+-------------------------------------------------------------------------------+
      | Method         | Description                                                                   |
      +================+===============================================================================+
      | :func:`match`  | returns whether the algorithm should be run given the current                 |
      |                | :class:`.Event` and :class:`.State`.                                          |
      +----------------+-------------------------------------------------------------------------------+
      | :func:`apply`  | Executes the algorithm's code and makes an in-place change                    |
      |                | to the :class:`.State`.                                                       |
      +----------------+-------------------------------------------------------------------------------+
    """

    def __init__(self, *args, **kwargs):  # Stub signature for PyRight
        del args, kwargs  # unused
        pass

    @property
    def find_unused_parameters(self) -> bool:
        """Indicates whether this algorithm may cause some model parameters to be unused. Defaults to False.

        For example, it is used to tell :class:`torch.nn.parallel.DistributedDataParallel` (DDP) that some parameters
        will be frozen during training, and hence it should not expect gradients from them. All algorithms which do any
        kind of parameter freezing should override this function to return ``True``.
        """
        return False

    @property
    def backwards_create_graph(self) -> bool:
        """Whether this algorithm requires the backwards pass to be differentiable. Defaults to ``False``.

        If it returns ``True``, ``create_graph=True`` will be passed to :meth:`torch.Tensor.backward` which will result in
        the graph of the gradient also being constructed. This allows the computation of second order derivatives.
        """
        return False

    @staticmethod
    def required_on_load() -> bool:
        """Return `True` to indicate this algorithm is required when loading from a checkpoint which used it."""
        return False

    def state_dict(self) -> dict[str, Any]:
        return {'repr': self.__repr__()}

    @abstractmethod
    def match(self, event: Event, state) -> bool:
        """Determines whether this algorithm should run given the current :class:`.Event` and :class:`.State`.

        Examples:
        To only run on a specific event (e.g., on :attr:`.Event.BEFORE_LOSS`), override match as shown below:

        >>> class MyAlgorithm:
        ...     def match(self, event, state):
        ...         return event == Event.BEFORE_LOSS
        >>> MyAlgorithm().match(Event.BEFORE_LOSS, state)
        True

        To run based on some value of a :class:`.State` attribute, override match as shown below:

        >>> class MyAlgorithm:
        ...     def match(self, event, state):
        ...        return state.timestamp.epoch > 30
        >>> MyAlgorithm().match(Event.BEFORE_LOSS, state)
        False

        See :class:`.State` for accessible attributes.

        Args:
            event (Event): The current event.
            state (State): The current state.

        Returns:
            bool: True if this algorithm should run now.
        """
        raise NotImplementedError(f'implement match() required for {self.__class__.__name__}')

    @abstractmethod
    def apply(self, event: Event, state, logger):
        """Applies the algorithm to make an in-place change to the :class:`.State`.

        Can optionally return an exit code to be stored in a :class:`.Trace`.
        This exit code is made accessible for debugging.

        Args:
            event (Event): The current event.
            state (State): The current state.
            logger (Logger): A logger to use for logging algorithm-specific metrics.

        Returns:
            int or None: exit code that will be stored in :class:`.Trace` and made accessible for debugging.
        """
        raise NotImplementedError(f'implement apply() required for {self.__class__.__name__}')
