from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MLPLaunchConfig:
    nnodes: int
    nproc_per_node: int
    node_rank: int
    master_addr: str
    master_port: int
    ifname: str | None = None

    def describe(self) -> str:
        return (
            f"nnodes={self.nnodes} nproc_per_node={self.nproc_per_node} "
            f"node_rank={self.node_rank} master_addr={self.master_addr} "
            f"master_port={self.master_port} ifname={self.ifname or 'auto'}"
        )


def infer_mlp_launch_config(
    env: Mapping[str, str] | None = None,
    *,
    master_port: int = 29630,
    nproc_per_node: int | None = None,
) -> MLPLaunchConfig:
    env = os.environ if env is None else env
    nnodes = _positive_int(_required(env, "MLP_WORKER_NUM"), "MLP_WORKER_NUM")
    node_rank = _nonnegative_int(_required(env, "MLP_ROLE_INDEX"), "MLP_ROLE_INDEX")
    if node_rank >= nnodes:
        raise ValueError(f"MLP_ROLE_INDEX={node_rank} must be smaller than MLP_WORKER_NUM={nnodes}.")

    if nproc_per_node is None:
        nproc_per_node = _positive_int(env.get("MLP_GPU", "1"), "MLP_GPU")
    else:
        nproc_per_node = _positive_int(str(nproc_per_node), "nproc_per_node")

    master_addr = env.get("MLP_WORKER_0_HOST") or _first_host(env.get("MLP_WORKER_ALL_HOSTS"))
    if not master_addr:
        raise RuntimeError("Cannot infer master address: MLP_WORKER_0_HOST and MLP_WORKER_ALL_HOSTS are missing.")

    return MLPLaunchConfig(
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=_positive_int(str(master_port), "master_port"),
        ifname=env.get("MLP_IFNAME") or env.get("NCCL_SOCKET_IFNAME"),
    )


def build_torchrun_command(
    config: MLPLaunchConfig,
    torchnep_args: Sequence[str],
    *,
    module: str = "wizard.torchNEP",
    python: str | None = None,
) -> list[str]:
    torchnep_args = _strip_remainder_separator(torchnep_args)
    if not torchnep_args:
        raise ValueError("mlp-launch requires a TorchNEP subcommand, e.g. train RUN_DIR.")
    python = sys.executable if python is None else python
    return [
        python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={config.nnodes}",
        f"--nproc_per_node={config.nproc_per_node}",
        f"--node_rank={config.node_rank}",
        f"--master_addr={config.master_addr}",
        f"--master_port={config.master_port}",
        "-m",
        module,
        *torchnep_args,
    ]


def build_launch_environment(config: MLPLaunchConfig, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["MASTER_ADDR"] = config.master_addr
    env["MASTER_PORT"] = str(config.master_port)
    env["NODE_RANK"] = str(config.node_rank)
    env["NNODES"] = str(config.nnodes)
    env["NPROC_PER_NODE"] = str(config.nproc_per_node)
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if config.ifname:
        env.setdefault("NCCL_SOCKET_IFNAME", config.ifname)
        env.setdefault("GLOO_SOCKET_IFNAME", config.ifname)
    return env


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_mlp_launch(
    torchnep_args: Sequence[str],
    *,
    master_port: int = 29630,
    nproc_per_node: int | None = None,
    dry_run: bool = False,
) -> int:
    torchnep_args = _strip_remainder_separator(torchnep_args)
    config = infer_mlp_launch_config(master_port=master_port, nproc_per_node=nproc_per_node)
    command = build_torchrun_command(config, torchnep_args)
    env = build_launch_environment(config)

    print(f"MLP launch: {config.describe()}", flush=True)
    print(f"Command: {format_command(command)}", flush=True)
    if dry_run:
        print("Dry run: command was not executed.", flush=True)
        return 0
    return subprocess.run(command, env=env, check=False).returncode


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} is required for MLP multi-node launch.")
    return value


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}.")
    return parsed


def _nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}.")
    return parsed


def _first_host(hosts: str | None) -> str | None:
    if not hosts:
        return None
    return hosts.split(",", 1)[0].strip() or None


def _strip_remainder_separator(args: Sequence[str]) -> list[str]:
    args = list(args)
    if args and args[0] == "--":
        return args[1:]
    return args
