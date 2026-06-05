from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from jsonargparse import ActionConfigFile, ArgumentParser

from holotrauma3x.arrival.cohort import TraumaCohort
from holotrauma3x.pathway import build_pathway
from holotrauma3x.protocol import Protocol, TriadInputs
from holotrauma3x.recovery.appraisal import Appraisal
from holotrauma3x.rounds.checkpoint import save_checkpoint
from holotrauma3x.rounds.devices import select_device
from holotrauma3x.rounds.logbook import get_logger
from holotrauma3x.rounds.loop import Trainer
from holotrauma3x.rounds.seeding import set_seed

_COMMANDS = ("train", "evaluate", "plan", "export")
_RUN_DIR = "runs"


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="holotrauma3x")
    subcommands = parser.add_subcommands()
    for name in _COMMANDS:
        child = ArgumentParser()
        child.add_argument("--config", action=ActionConfigFile)
        child.add_dataclass_arguments(Protocol, "protocol")
        subcommands.add_subcommand(name, child)
    return parser


def _cohort(protocol: Protocol) -> TraumaCohort:
    return TraumaCohort(
        protocol.arrival,
        protocol.arrival.cohort_size,
        protocol.training.seed,
        protocol.training.centers,
    )


def run_train(protocol: Protocol) -> None:
    logger = get_logger("holotrauma3x.train")
    set_seed(protocol.training.seed)
    model = build_pathway(protocol)
    history = Trainer(protocol.training).fit(model, _cohort(protocol))
    logger.info(
        "experiment=%s steps=%d final_loss=%.6f",
        protocol.experiment,
        len(history.step_losses),
        history.final,
    )
    os.makedirs(_RUN_DIR, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(
        os.path.join(_RUN_DIR, f"{protocol.experiment}.pt"),
        model,
        optimizer,
        protocol.training.seed,
        history.epochs,
    )


def run_evaluate(protocol: Protocol) -> None:
    logger = get_logger("holotrauma3x.evaluate")
    set_seed(protocol.training.seed)
    model = build_pathway(protocol)
    metrics = Appraisal(
        select_device(protocol.training.device), protocol.optimization.airway_critical
    ).evaluate(model, _cohort(protocol), protocol.training.batch_size)
    for key, value in metrics.items():
        logger.info("%s=%.4f", key, value)


def run_plan(protocol: Protocol) -> None:
    logger = get_logger("holotrauma3x.plan")
    set_seed(protocol.training.seed)
    model = build_pathway(protocol)
    sample = _cohort(protocol)[0]
    inputs = TriadInputs(
        volume=sample["volume"].unsqueeze(0),
        points=sample["points"].unsqueeze(0),
        radiograph=sample["radiograph"].unsqueeze(0),
    )
    plan = model.plan(inputs, seed=protocol.training.seed)
    logger.info("confidence=%.3f", float(plan.confidence[0].item()))
    logger.info("trajectory_waypoints=%d", plan.trajectory.shape[0])
    logger.info("reasoning=%s", plan.narrative[0] if plan.narrative else "")


def run_export(protocol: Protocol) -> None:
    logger = get_logger("holotrauma3x.export")
    set_seed(protocol.training.seed)
    model = build_pathway(protocol)
    os.makedirs(_RUN_DIR, exist_ok=True)
    target = os.path.join(_RUN_DIR, f"{protocol.experiment}_state.pt")
    torch.save(model.state_dict(), target)
    logger.info("saved=%s", target)


def main(args: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    config = parser.parse_args(args)
    instantiated = parser.instantiate(config)
    command = instantiated.subcommand
    protocol = getattr(instantiated, command).protocol
    dispatch = {
        "train": run_train,
        "evaluate": run_evaluate,
        "plan": run_plan,
        "export": run_export,
    }
    dispatch[command](protocol)


if __name__ == "__main__":
    main()
