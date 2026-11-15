"""Real-time training visibility for the relative-action training loop.

Wraps ``rich.live.Live`` to surface a single in-terminal panel that
auto-updates after every episode and every evaluation checkpoint. Also
writes a tailable ``live_status.json`` next to the run's CSV logs so the
run can be monitored with ``Get-Content -Wait`` or ``tail -f`` even when
the terminal isn't attached (e.g. over SSH or redirected to a file).

The module is deliberately additive: it never touches the CSV / TensorBoard
writes in :mod:`train.run_logger`. If ``rich`` isn't installed or stdout
isn't a TTY, the class degrades to a no-op renderer that still writes the
tailable JSON and falls through to whatever ``print(...)`` the caller
wants to use.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from typing import Any, Iterable, Mapping


_RICH_AVAILABLE = True
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover - import guard
    _RICH_AVAILABLE = False


# Sparkline glyphs, ordered from lowest to highest.
_SPARK_GLYPHS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: Iterable[float]) -> str:
    """Render an inline bar-chart of *values* using block-drawing glyphs."""
    series = [v for v in values if v is not None]
    if not series:
        return ""
    lo, hi = min(series), max(series)
    if hi == lo:
        glyph = _SPARK_GLYPHS[len(_SPARK_GLYPHS) // 2]
        return glyph * len(series)
    spread = hi - lo
    out = []
    for v in series:
        idx = int((v - lo) / spread * (len(_SPARK_GLYPHS) - 1))
        idx = max(0, min(len(_SPARK_GLYPHS) - 1, idx))
        out.append(_SPARK_GLYPHS[idx])
    return "".join(out)


def _safe_mean(values: Iterable[float]) -> float:
    series = [v for v in values if v is not None]
    return sum(series) / len(series) if series else 0.0


def _safe_max(values: Iterable[float]) -> float:
    series = [v for v in values if v is not None]
    return max(series) if series else 0.0


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_signed(value: float, fmt: str = "+.1f") -> str:
    """Format a float with an explicit sign so deltas line up in columns."""
    return format(value, fmt)


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class EpisodeProgress:
    """Stateful, real-time training monitor.

    Parameters
    ----------
    n_episodes:
        Total number of episodes the run will execute.
    run_dir:
        Directory where ``live_status.json`` is written. Will be created
        if missing.
    run_tag:
        Run identifier embedded in the JSON payload and panel header.
    env_cfg:
        Subset of env hyperparameters to surface in the panel header
        (e.g. ``device``, ``max_dispatch_slots``, ``traffic_mode``).
    rolling_window:
        Episode count used for the rolling reward / destruction /
        containment stats. Defaults to 50.
    eval_scenarios:
        Ordered iterable of scenario names (e.g.
        ``["anchor", "mandeville", "getty"]``). Used to render the
        evaluation table with stable columns.
    enable:
        ``False`` to disable the rich panel even when both ``rich`` and
        a TTY are present. The tail-file writer still runs.
    """

    def __init__(
        self,
        n_episodes_start: int,
        n_episodes: int,
        run_dir: str,
        run_tag: str,
        env_cfg: Mapping[str, Any] | None = None,
        rolling_window: int = 50,
        eval_scenarios: Iterable[str] = (),
        enable: bool = True,
    ):
        self.n_episodes_start = int(n_episodes_start)
        self.n_episodes = int(n_episodes)
        self.run_dir = run_dir
        self.run_tag = run_tag
        self.env_cfg = dict(env_cfg or {})
        self.rolling_window = max(1, int(rolling_window))
        self.eval_scenarios = list(eval_scenarios)
        self.enable_rich = bool(enable) and _RICH_AVAILABLE and _is_tty()

        os.makedirs(self.run_dir, exist_ok=True)
        self.status_path = os.path.join(self.run_dir, "live_status.json")

        # Per-episode rolling buffers (sized to rolling_window).
        self._reward_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._destroyed_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._contained_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._ticks_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._entropy_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._grad_norm_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._policy_loss_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._value_loss_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._noop_buf: deque[float] = deque(maxlen=self.rolling_window)
        self._trench_bonus_buf: deque[float] = deque(maxlen=self.rolling_window)
        # Sparkline uses the rolling window by default.
        self._spark_buf: deque[float] = deque(maxlen=max(self.rolling_window, 80))

        # Eval history: list of {episode, scenario, reward, destroyed, containment}.
        self._eval_history: list[dict[str, Any]] = []
        # Last eval value per scenario, used for delta computation.
        self._last_eval: dict[str, dict[str, Any]] = {}

        # Timing.
        self._t_start = time.monotonic()
        self._t_last_episode = self._t_start
        # Most recent episode summary (persists into the tail file on close()).
        self._latest_summary: dict[str, Any] | None = None

        # Rich state.
        self._console: Console | None = None
        self._live: Live | None = None
        if self.enable_rich:
            try:
                self._console = Console()
                self._live = Live(
                    self._build_panel(),
                    console=self._console,
                    refresh_per_second=8,
                    transient=False,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
            except Exception:
                self._console = None
                self._live = None
                self.enable_rich = False

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    @property
    def is_live(self) -> bool:
        """True when the rich panel is actively rendering."""
        return self._live is not None

    @property
    def status_file(self) -> str:
        return self.status_path

    def __enter__(self) -> "EpisodeProgress":
        if self._live is not None:
            self._live.start()
        # Write an initial status file so tail consumers see the run tag.
        self._write_status()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        # Final flush so the tail file reflects the last observed episode.
        self._write_status()

    def tick(self, episode: int, summary: Mapping[str, Any]) -> None:
        """Record an episode result and refresh the panel / tail file."""
        ep = int(episode)
        reward = float(summary.get("total_reward", 0.0) or 0.0)
        destroyed = float(summary.get("buildings_destroyed", 0) or 0)
        contained = float(bool(summary.get("contained", False)))
        n_ticks = float(summary.get("n_ticks", 0) or 0)
        entropy = float(summary.get("entropy", 0.0) or 0.0)
        grad_norm = summary.get("grad_norm", "")
        policy_loss = float(summary.get("policy_loss", 0.0) or 0.0)
        value_loss = float(summary.get("value_loss", 0.0) or 0.0)
        noop = float(summary.get("noop_ticks", 0) or 0)
        trench_bonus = float(summary.get("rc_trench_bonus", 0.0) or 0.0)

        self._reward_buf.append(reward)
        self._destroyed_buf.append(destroyed)
        self._contained_buf.append(contained)
        self._ticks_buf.append(n_ticks)
        self._entropy_buf.append(entropy)
        self._policy_loss_buf.append(policy_loss)
        self._value_loss_buf.append(value_loss)
        self._noop_buf.append(noop)
        self._trench_bonus_buf.append(trench_bonus)
        if grad_norm != "" and grad_norm is not None:
            try:
                self._grad_norm_buf.append(float(grad_norm))
            except (TypeError, ValueError):
                pass
        self._spark_buf.append(reward)

        self._t_last_episode = time.monotonic()
        self._latest_summary = dict(summary)
        self._write_status(latest_episode=ep, latest_summary=self._latest_summary)
        self._refresh()

    def refresh_evals(self, eval_rows: Iterable[Mapping[str, Any]]) -> None:
        """Update the eval-history table after a checkpoint evaluation.

        *eval_rows* is an iterable of dicts shaped like the ``EVAL_FIELDS``
        row written by :class:`RunLogger` (plus whatever extras the caller
        passes - only ``episode``, ``scenario``, ``avg_reward``,
        ``avg_buildings_destroyed`` and ``containment_rate`` are displayed).
        """
        for row in eval_rows:
            scenario = str(row.get("scenario", "?"))
            entry = {
                "episode": int(row.get("checkpoint_episode", row.get("episode", 0))),
                "scenario": scenario,
                "reward": float(row.get("avg_reward", 0.0) or 0.0),
                "destroyed": float(row.get("avg_buildings_destroyed", 0.0) or 0.0),
                "containment": float(row.get("containment_rate", 0.0) or 0.0),
            }
            self._eval_history.append(entry)
            self._last_eval[scenario] = entry
        self._write_status()
        self._refresh()

    # ------------------------------------------------------------------ #
    # Status-file writer (atomic, JSON)
    # ------------------------------------------------------------------ #
    def _write_status(
        self,
        latest_episode: int | None = None,
        latest_summary: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write ``live_status.json`` for tail consumers."""
        elapsed = max(1e-6, time.monotonic() - self._t_start)
        eps_per_sec = (
            (latest_episode / elapsed) if latest_episode else 0.0
        )
        remaining_eps = max(0, self.n_episodes - (latest_episode or 0))
        eta_seconds = (remaining_eps / eps_per_sec) if eps_per_sec > 0 else 0.0

        # Default to the cached latest summary so close() and refresh_evals()
        # never wipe the "last" key from the tail file.
        effective_summary = latest_summary if latest_summary is not None else self._latest_summary

        payload: dict[str, Any] = {
            "run_tag": self.run_tag,
            "updated_at": time.time(),
            "n_episodes": self.n_episodes,
            "last_episode": latest_episode,
            "elapsed_s": elapsed,
            "eps_per_sec": eps_per_sec,
            "eta_s": eta_seconds,
            "env": self.env_cfg,
            "rolling_window": self.rolling_window,
            "rolling": {
                "reward_mean": _safe_mean(self._reward_buf),
                "reward_min": _safe_min(self._reward_buf),
                "reward_max": _safe_max(self._reward_buf),
                "destroyed_mean": _safe_mean(self._destroyed_buf),
                "containment_rate": _safe_mean(self._contained_buf),
                "ticks_mean": _safe_mean(self._ticks_buf),
                "entropy_mean": _safe_mean(self._entropy_buf),
                "grad_norm_mean": _safe_mean(self._grad_norm_buf),
                "policy_loss_mean": _safe_mean(self._policy_loss_buf),
                "value_loss_mean": _safe_mean(self._value_loss_buf),
                "noop_ticks_mean": _safe_mean(self._noop_buf),
                "trench_bonus_mean": _safe_mean(self._trench_bonus_buf),
            },
            "spark_recent_rewards": list(self._spark_buf),
            "eval_history": list(self._eval_history),
        }
        if effective_summary is not None:
            payload["last"] = dict(effective_summary)

        tmp_path = self.status_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, self.status_path)
        except Exception:
            # Best-effort: never raise from logging.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Rich rendering
    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(self._build_panel())
        except Exception:
            pass

    def _build_panel(self) -> "Panel | Text":
        return _build_panel(
            run_tag=self.run_tag,
            n_episodes=self.n_episodes,
            env_cfg=self.env_cfg,
            t_start=self._t_start,
            reward_buf=self._reward_buf,
            destroyed_buf=self._destroyed_buf,
            contained_buf=self._contained_buf,
            entropy_buf=self._entropy_buf,
            grad_norm_buf=self._grad_norm_buf,
            noop_buf=self._noop_buf,
            spark_buf=self._spark_buf,
            eval_history=self._eval_history,
            eval_scenarios=self.eval_scenarios,
            last_eval=self._last_eval,
        )


def _safe_min(values: Iterable[float]) -> float:
    series = [v for v in values if v is not None]
    return min(series) if series else 0.0


# ---------------------------------------------------------------------- #
# Module-level rendering helpers (kept separate so they're easy to test)
# ---------------------------------------------------------------------- #
def _build_panel(
    *,
    run_tag: str,
    n_episodes: int,
    env_cfg: Mapping[str, Any],
    t_start: float,
    reward_buf: Iterable[float],
    destroyed_buf: Iterable[float],
    contained_buf: Iterable[float],
    entropy_buf: Iterable[float],
    grad_norm_buf: Iterable[float],
    noop_buf: Iterable[float],
    spark_buf: Iterable[float],
    eval_history: list[Mapping[str, Any]],
    eval_scenarios: Iterable[str],
    last_eval: Mapping[str, dict[str, Any]],
) -> Any:
    if not _RICH_AVAILABLE:  # pragma: no cover - import guard
        return None
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    elapsed = max(0.0, time.monotonic() - t_start)
    cur_ep = len(reward_buf) if hasattr(reward_buf, "__len__") else 0
    # reward_buf is a deque - use its current length for "current episode".
    cur_ep = len(list(reward_buf)) if reward_buf else 0
    pct = (cur_ep / n_episodes) if n_episodes else 0.0
    bar_width = 30
    filled = int(round(pct * bar_width))
    bar = "█" * filled + "·" * (bar_width - filled)
    eps_per_sec = cur_ep / elapsed if elapsed > 0 else 0.0
    remaining = max(0, n_episodes - cur_ep)
    eta = (remaining / eps_per_sec) if eps_per_sec > 0 else 0.0

    spark = _sparkline(spark_buf)
    reward_mean = _safe_mean(reward_buf)
    destroyed_mean = _safe_mean(destroyed_buf)
    contain_pct = _safe_mean(contained_buf) * 100
    entropy_mean = _safe_mean(entropy_buf)
    grad_mean = _safe_mean(grad_norm_buf)
    noop_mean = _safe_mean(noop_buf)
    reward_min = _safe_min(reward_buf)
    reward_max = _safe_max(reward_buf)

    header = Text()
    header.append(f"{run_tag}\n", style="bold cyan")
    header.append(f"  [{bar}] {cur_ep}/{n_episodes} ", style="green")
    header.append(f"({pct * 100:5.1f}%) ", style="dim")
    header.append(
        f"ETA {_format_seconds(eta)}  {eps_per_sec:.2f} eps/s  "
        f"elapsed {_format_seconds(elapsed)}\n",
        style="yellow",
    )
    env_bits = []
    if "device" in env_cfg:
        env_bits.append(f"device={env_cfg['device']}")
    if "max_dispatch_slots" in env_cfg:
        env_bits.append(f"slots={env_cfg['max_dispatch_slots']}")
    if "traffic_mode" in env_cfg:
        env_bits.append(f"traffic={env_cfg['traffic_mode']}")
    if "learning_rate" in env_cfg:
        env_bits.append(f"lr={env_cfg['learning_rate']}")
    if env_bits:
        header.append("  " + "  ".join(env_bits), style="dim")

    rolling = Table(
        title=f"Rolling (last {len(list(reward_buf)) or '?'} episodes)",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    rolling.add_column("Metric", style="bold")
    rolling.add_column("Mean", justify="right")
    rolling.add_column("Min", justify="right")
    rolling.add_column("Max", justify="right")
    rolling.add_row(
        "reward",
        f"{reward_mean:+.1f}",
        f"{reward_min:+.1f}",
        f"{reward_max:+.1f}",
    )
    rolling.add_row(
        "destroyed",
        f"{destroyed_mean:.2f}",
        "-",
        "-",
    )
    rolling.add_row(
        "containment %",
        f"{contain_pct:5.1f}",
        "-",
        "-",
    )
    rolling.add_row(
        "entropy",
        f"{entropy_mean:.3f}",
        "-",
        "-",
    )
    if grad_norm_buf:
        rolling.add_row(
            "grad_norm",
            f"{grad_mean:.3f}",
            "-",
            "-",
        )
    rolling.add_row(
        "noop_ticks",
        f"{noop_mean:.2f}",
        "-",
        "-",
    )
    rolling.add_row(
        "reward spark",
        spark or "(waiting for data…)",
        "-",
        "-",
    )

    # Build eval table.
    eval_tbl = Table(
        title="Evaluation (Δ vs previous checkpoint)",
        show_header=True,
        header_style="bold blue",
        expand=True,
    )
    eval_tbl.add_column("Episode", justify="right")
    eval_tbl.add_column("Scenario", style="bold")
    eval_tbl.add_column("Reward", justify="right")
    eval_tbl.add_column("Δ reward", justify="right")
    eval_tbl.add_column("Destroyed", justify="right")
    eval_tbl.add_column("Δ destroyed", justify="right")
    eval_tbl.add_column("Contain", justify="right")

    scenarios = list(eval_scenarios) or sorted({h["scenario"] for h in eval_history})
    if not eval_history:
        eval_tbl.add_row("-", "(no evaluations yet)", "-", "-", "-", "-", "-")
    else:
        # Show last occurrence per scenario for the most recent checkpoint
        # episode, plus the prior one for delta. We group by checkpoint.
        by_episode: dict[int, list[dict[str, Any]]] = {}
        for h in eval_history:
            by_episode.setdefault(int(h["episode"]), []).append(dict(h))
        ep_keys = sorted(by_episode.keys())
        # Most recent eval block per scenario:
        last_per_scenario: dict[str, dict[str, Any]] = {}
        prev_per_scenario: dict[str, dict[str, Any]] = {}
        if ep_keys:
            latest_ep = ep_keys[-1]
            latest_rows = by_episode[latest_ep]
            if len(ep_keys) >= 2:
                prev_rows = by_episode[ep_keys[-2]]
            else:
                prev_rows = []
            prev_by_scen = {r["scenario"]: r for r in prev_rows}
            for r in latest_rows:
                scen = r["scenario"]
                last_per_scenario[scen] = r
                prev_per_scenario[scen] = prev_by_scen.get(scen)

        ordered = [s for s in scenarios if s in last_per_scenario] + [
            s for s in last_per_scenario if s not in scenarios
        ]
        for scen in ordered:
            r = last_per_scenario[scen]
            p = prev_per_scenario.get(scen)
            d_reward = (r["reward"] - p["reward"]) if p else 0.0
            d_destroyed = (r["destroyed"] - p["destroyed"]) if p else 0.0
            reward_style = _delta_style(d_reward)
            destroy_style = _delta_style(-d_destroyed)  # less destroyed is better
            eval_tbl.add_row(
                str(r["episode"]),
                scen,
                f"{r['reward']:+.1f}",
                Text(_format_signed(d_reward), style=reward_style),
                f"{r['destroyed']:.2f}",
                Text(_format_signed(d_destroyed), style=destroy_style),
                f"{r['containment'] * 100:5.1f}%",
            )
        if not last_per_scenario:
            eval_tbl.add_row("-", "(no evaluations yet)", "-", "-", "-", "-", "-")

    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(header)
    body.add_row(rolling)
    body.add_row(eval_tbl)
    return Panel(body, title="Inferno Tactics · relative model training", border_style="cyan")


def _delta_style(value: float) -> str:
    """Pick a rich style string for a delta value.

    Positive reward delta and negative destroyed delta are "good";
    everything else is "neutral" or "bad".
    """
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "dim"
