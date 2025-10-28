"""
Reusable plotting utilities for MDP solver outputs.

These functions encapsulate the plotting logic currently shared between the
solver and simulator Quarto documents to ensure consistent visualizations.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_convergence_history(
    *,
    iterations: Sequence[int],
    max_errors: Sequence[float],
    epsilon_tol: float,
    ax: plt.Axes | None = None,
    title: str = "Convergence of Value Function Iteration",
) -> plt.Axes:
    """
    Plot convergence history of the value iteration process.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, max_errors, 'b-', linewidth=2, label='Max Error')
    ax.axhline(y=epsilon_tol, color='r', linestyle='--', linewidth=1.5, label=f'Tolerance (ε={epsilon_tol})')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Maximum Error', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    return ax


def plot_choice_value_functions(
    *,
    state_grid: Sequence[float],
    v0_values: Sequence[float],
    v1_values: Sequence[float],
    ax: plt.Axes | None = None,
    title: str = "Choice-Specific Value Functions",
) -> plt.Axes:
    """
    Plot choice-specific value functions on a shared axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(state_grid, v0_values, 'b-', linewidth=2.5, label='v(s, a=0)', alpha=0.8)
    ax.plot(state_grid, v1_values, 'r-', linewidth=2.5, label='v(s, a=1)', alpha=0.8)

    ax.set_xlabel('State (s)', fontsize=12)
    ax.set_ylabel('Value Function v(s, a)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    return ax


def plot_policy_probabilities(
    *,
    state_grid: Sequence[float],
    prob_a0: Sequence[float],
    prob_a1: Sequence[float],
    ax: plt.Axes | None = None,
    title: str = "Optimal Policy (Choice Probabilities)",
) -> plt.Axes:
    """
    Plot optimal policy choice probabilities.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(state_grid, prob_a0, 'b-', linewidth=2.5, label='P(a=0|s)', alpha=0.8)
    ax.plot(state_grid, prob_a1, 'r-', linewidth=2.5, label='P(a=1|s)', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('State (s)', fontsize=12)
    ax.set_ylabel('Choice Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    return ax


def _mix_with_color(base_color: Sequence[float], mix_color: str, weight: float) -> tuple[float, float, float, float]:
    """
    Blend grayscale base color with a target chromatic color.
    """
    mix_color = mix_color.lower()
    if mix_color == 'blue':
        return (
            base_color[0] * (1 - weight),
            base_color[1] * (1 - weight),
            base_color[2] * (1 - weight) + weight,
            1,
        )
    if mix_color == 'red':
        return (
            base_color[0] * (1 - weight) + weight,
            base_color[1] * (1 - weight),
            base_color[2] * (1 - weight),
            1,
        )
    if mix_color == 'purple':
        return (
            base_color[0] * (1 - weight) + weight * 0.6,
            base_color[1] * (1 - weight),
            base_color[2] * (1 - weight) + weight,
            1,
        )
    if mix_color == 'orange':
        return (
            base_color[0] * (1 - weight) + weight,
            base_color[1] * (1 - weight) + weight * 0.5,
            base_color[2] * (1 - weight),
            1,
        )
    return base_color


def _plot_parameterized_series(
    *,
    ax: plt.Axes,
    state_grid: Sequence[float],
    results: Iterable[dict],
    value_key: str,
    parameter_key: str,
    mix_color: str,
    ylabel: str,
    title: str,
    label_interval: float = 0.5,
) -> None:
    labels = [result[parameter_key] for result in results]
    min_label = min(labels)
    max_label = max(labels)

    cmap = cm.get_cmap('Greys_r')

    for result in results:
        label = result[parameter_key]
        if max_label > min_label:
            weight = (label - min_label) / (max_label - min_label)
        else:
            weight = 0.5

        base_color = cmap(weight)
        color = _mix_with_color(base_color, mix_color=mix_color, weight=weight)

        if label_interval > 0:
            scaled = label / label_interval
            label_text = f'{parameter_key}={label:.2f}' if abs(scaled - round(scaled)) < 1e-8 else ''
        else:
            label_text = ''
        ax.plot(state_grid, result[value_key], linewidth=2, alpha=0.7, color=color, label=label_text)

    ax.set_xlabel('State (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_comparative_value_functions(
    *,
    state_grid: Sequence[float],
    results: Iterable[dict],
    parameter_key: str,
    label_interval: float = 0.5,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot comparative statics for value functions across parameter values.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))

    _plot_parameterized_series(
        ax=ax0,
        state_grid=state_grid,
        results=results,
        value_key='v0',
        parameter_key=parameter_key,
        mix_color='blue',
        ylabel='Value Function v(s, a=0)',
        title=f'Value Functions for a=0 (Black→Blue as {parameter_key} increases)',
        label_interval=label_interval,
    )

    _plot_parameterized_series(
        ax=ax1,
        state_grid=state_grid,
        results=results,
        value_key='v1',
        parameter_key=parameter_key,
        mix_color='red',
        ylabel='Value Function v(s, a=1)',
        title=f'Value Functions for a=1 (Black→Red as {parameter_key} increases)',
        label_interval=label_interval,
    )

    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_comparative_policies(
    *,
    state_grid: Sequence[float],
    results: Iterable[dict],
    parameter_key: str,
    label_interval: float = 0.5,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot comparative statics for policy probabilities across parameter values.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))

    _plot_parameterized_series(
        ax=ax0,
        state_grid=state_grid,
        results=results,
        value_key='prob_a0',
        parameter_key=parameter_key,
        mix_color='blue',
        ylabel='P(a=0|s)',
        title=f'Policy for a=0 (Black→Blue as {parameter_key} increases)',
        label_interval=label_interval,
    )
    ax0.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax0.set_ylim([0, 1])

    _plot_parameterized_series(
        ax=ax1,
        state_grid=state_grid,
        results=results,
        value_key='prob_a1',
        parameter_key=parameter_key,
        mix_color='red',
        ylabel='P(a=1|s)',
        title=f'Policy for a=1 (Black→Red as {parameter_key} increases)',
        label_interval=label_interval,
    )
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_ylim([0, 1])

    fig.tight_layout()
    return fig, (ax0, ax1)
