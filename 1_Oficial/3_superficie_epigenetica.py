#!/usr/bin/env python3
"""Etapa 3 - Constrói a superfície epigenética (quasi-potencial) a partir do landscape. Usa MPI."""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpi4py import MPI
import yaml


# =========================================================
# MPI
# =========================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# =========================================================
# FUNÇÕES AUXILIARES DE CONFIGURAÇÃO
# =========================================================


def fail_config(message):
    raise ValueError(f"Erro no arquivo de configuração: {message}")


def require_mapping(config, key):
    value = config.get(key)
    if not isinstance(value, dict):
        fail_config(f"seção obrigatória '{key}' ausente ou inválida.")
    return value


def resolve_config_path(value, config_dir):
    """Resolve caminhos do YAML relativos à pasta do config.yaml."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def resolve_runtime_path(value):
    """Resolve caminhos da linha de comando relativos ao diretório atual."""
    return Path(str(value)).expanduser().resolve()


def safe_filename_component(text):
    """Nome de arquivo seguro para as saídas automáticas."""
    text = str(text)
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


# =========================================================
# ARGUMENTOS
# =========================================================

parser = argparse.ArgumentParser(
    description=(
        "Constrói a superfície epigenética a partir do landscape numérico, "
        "usando a ordem das colunas definida em report.species do config.yaml."
    )
)
parser.add_argument("config", help="Arquivo YAML de configuração.")
parser.add_argument(
    "--landscape",
    help="Sobrescreve output.landscape.",
)
parser.add_argument(
    "--matrix",
    help="Sobrescreve output.matrix.",
)
parser.add_argument(
    "--figure",
    help="Sobrescreve output.figure.",
)
args = parser.parse_args()

config_path = Path(args.config).expanduser().resolve()
if not config_path.is_file():
    raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

with config_path.open("r", encoding="utf-8") as fh:
    config = yaml.safe_load(fh)

if not isinstance(config, dict):
    fail_config("o conteúdo principal do YAML deve ser um mapeamento/chave-valor.")

config_dir = config_path.parent


# =========================================================
# LANDSCAPE DE ENTRADA
# =========================================================

output_cfg = require_mapping(config, "output")

if args.landscape:
    filepath = Path(args.landscape).expanduser().resolve()
else:
    if not output_cfg.get("landscape"):
        fail_config("informe 'output.landscape'.")
    filepath = resolve_config_path(output_cfg["landscape"], config_dir)

if not filepath.is_file():
    raise FileNotFoundError(f"Arquivo landscape não encontrado: {filepath}")


# =========================================================
# ORDEM DAS COLUNAS DO REPORT
# =========================================================

report_cfg = require_mapping(config, "report")
report_species = report_cfg.get("species")

if not isinstance(report_species, list) or len(report_species) == 0:
    fail_config("'report.species' deve ser uma lista não vazia e ordenada.")

report_species = [str(x).strip() for x in report_species]

if any(not name for name in report_species):
    fail_config("'report.species' contém um nome vazio.")

if len(set(report_species)) != len(report_species):
    fail_config("existem espécies repetidas em 'report.species'.")

if any(name.lower() == "time" for name in report_species):
    fail_config(
        "não inclua 'Time' em 'report.species': Time é automaticamente a coluna 0."
    )

# Colunas do landscape: 0 = Time, depois report.species na ordem
column_map = {"Time": 0}
column_map.update({name: i for i, name in enumerate(report_species, start=1)})


# =========================================================
# CONFIGURAÇÃO DA SUPERFÍCIE
# =========================================================

plot_cfg = require_mapping(config, "landscape_plot")

x_species = str(plot_cfg.get("x_species", "")).strip()
y_species = str(plot_cfg.get("y_species", "")).strip()

if not x_species:
    fail_config("informe 'landscape_plot.x_species'.")
if not y_species:
    fail_config("informe 'landscape_plot.y_species'.")
if x_species == y_species:
    fail_config("x_species e y_species devem ser espécies diferentes.")

if x_species not in column_map:
    fail_config(
        f"a espécie X '{x_species}' não existe em 'report.species'. "
        f"Disponíveis: {', '.join(report_species)}"
    )

if y_species not in column_map:
    fail_config(
        f"a espécie Y '{y_species}' não existe em 'report.species'. "
        f"Disponíveis: {', '.join(report_species)}"
    )

col_X = column_map[x_species]
col_Y = column_map[y_species]

try:
    dim_matriz = int(plot_cfg.get("matrix_dimension"))
except (TypeError, ValueError):
    fail_config("'landscape_plot.matrix_dimension' deve ser um número inteiro.")

if dim_matriz < 2:
    fail_config("'landscape_plot.matrix_dimension' deve ser >= 2.")


# =========================================================
# ARQUIVOS DE SAÍDA DO DRAW LANDSCAPE
# =========================================================

safe_x = safe_filename_component(x_species)
safe_y = safe_filename_component(y_species)

matrix_value = output_cfg.get(
    "matrix",
    f"SavedFileMatrizC_{safe_x}_{safe_y}.npz",
)
figure_value = output_cfg.get(
    "figure",
    f"landscape_{safe_x}_{safe_y}.svg",
)

if args.matrix:
    matrix_path = resolve_runtime_path(args.matrix)
else:
    matrix_path = resolve_config_path(matrix_value, config_dir)

if args.figure:
    figure_path = resolve_runtime_path(args.figure)
else:
    figure_path = resolve_config_path(figure_value, config_dir)

matrix_path.parent.mkdir(parents=True, exist_ok=True)
figure_path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# DISTRIBUIÇÃO DO ARQUIVO ENTRE PROCESSOS MPI
# =========================================================

file_size = os.path.getsize(filepath)
if file_size == 0:
    raise ValueError(f"O arquivo landscape está vazio: {filepath}")

chunk_size = file_size // size
start_byte = rank * chunk_size
end_byte = (rank + 1) * chunk_size if rank != size - 1 else file_size


def read_chunk_lines(path, start, end):
    """Lê as linhas do intervalo de bytes deste rank (a linha parcial inicial fica com o rank anterior)."""
    with open(path, "rb") as f:
        f.seek(start)

        if start != 0:
            f.readline()

        line = f.readline()
        while line:
            yield line.decode("ascii", errors="ignore")

            if f.tell() >= end:
                break

            line = f.readline()


# =========================================================
# LEITURA DOS LIMITES DAS DUAS ESPÉCIES
# =========================================================


def find_max_min_any_2_species(path, col_x, col_y):
    local_x_max = float("-inf")
    local_y_max = float("-inf")
    local_x_min = float("inf")
    local_y_min = float("inf")
    valid_rows = 0

    for line in read_chunk_lines(path, start_byte, end_byte):
        parts = line.strip().split()

        if len(parts) <= max(col_x, col_y):
            continue

        try:
            x_value = float(parts[col_x])
            y_value = float(parts[col_y])
        except ValueError:
            continue

        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue

        local_x_max = max(local_x_max, x_value)
        local_x_min = min(local_x_min, x_value)
        local_y_max = max(local_y_max, y_value)
        local_y_min = min(local_y_min, y_value)
        valid_rows += 1

    return local_x_max, local_x_min, local_y_max, local_y_min, valid_rows


# =========================================================
# MATRIZ DE CONTAGEM
# =========================================================


def calc_matriz_cont_any_2_species(
    path,
    col_x,
    col_y,
    x_max,
    x_min,
    y_max,
    y_min,
):
    matriz_cont_local = np.zeros((dim_matriz, dim_matriz))
    points_total_local = 0

    x_bin_width = (x_max - x_min) / dim_matriz
    y_bin_width = (y_max - y_min) / dim_matriz

    for line in read_chunk_lines(path, start_byte, end_byte):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        parts = stripped_line.split()
        if len(parts) <= max(col_x, col_y):
            continue

        try:
            x_value = float(parts[col_x])
            y_value = float(parts[col_y])
        except ValueError:
            continue

        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue

        op_x = int((x_value - x_min) / x_bin_width) - 1
        op_y = int((y_value - y_min) / y_bin_width) - 1

        op_x = min(max(op_x, 0), dim_matriz - 1)
        op_y = min(max(op_y, 0), dim_matriz - 1)

        matriz_cont_local[op_x, op_y] += 1
        points_total_local += 1

    return matriz_cont_local, points_total_local


# =========================================================
# PROBABILIDADE E QUASI-POTENCIAL
# =========================================================


def calc_matriz_prob(matriz_cont, total_points):
    return matriz_cont / total_points if total_points > 0 else matriz_cont


def calc_matriz_inv_log(matriz_prob):
    vetor_z = np.zeros_like(matriz_prob)
    vetor_z[matriz_prob != 0] = -np.log(matriz_prob[matriz_prob != 0])
    return np.ma.masked_where(matriz_prob == 0, vetor_z)


# =========================================================
# DESENHO DA SUPERFÍCIE
# =========================================================


def draw_surface_3d_matplotlib(
    vetor_z_masked,
    x_max,
    x_min,
    y_max,
    y_min,
    dim,
    label_x,
    label_y,
    output_file,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    x_axis = np.linspace(x_min, x_max, dim)
    y_axis = np.linspace(y_min, y_max, dim)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    surf = ax.plot_surface(
        x_mesh,
        y_mesh,
        vetor_z_masked,
        cmap=cm.plasma,
        linewidth=0,
        antialiased=False,
    )

    ax.set_xlabel(label_x, fontsize=12, labelpad=10)
    ax.set_ylabel(label_y, fontsize=12, labelpad=10)
    ax.set_zlabel("Quasi-potencial", fontsize=12, labelpad=10)
    ax.view_init(elev=25, azim=636)

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(output_file)
    plt.close(fig)


# =========================================================
# EXECUÇÃO
# =========================================================

start = MPI.Wtime()

if rank == 0:
    print("==============================================")
    print("DRAW LANDSCAPE")
    print("Landscape:", filepath)
    print(f"Eixo X: {x_species} -> coluna {col_X}")
    print(f"Eixo Y: {y_species} -> coluna {col_Y}")
    print("Dimensão da matriz:", dim_matriz)
    print("Processos MPI:", size)
    print("==============================================")

local_x_max, local_x_min, local_y_max, local_y_min, local_valid_rows = (
    find_max_min_any_2_species(filepath, col_X, col_Y)
)

valid_rows_total = comm.allreduce(local_valid_rows, op=MPI.SUM)

if valid_rows_total == 0:
    raise ValueError(
        "Nenhuma linha numérica válida foi encontrada nas colunas selecionadas. "
        "Verifique report.species e o arquivo landscape."
    )

x_max = comm.allreduce(local_x_max, op=MPI.MAX)
x_min = comm.allreduce(local_x_min, op=MPI.MIN)
y_max = comm.allreduce(local_y_max, op=MPI.MAX)
y_min = comm.allreduce(local_y_min, op=MPI.MIN)

if x_max == x_min:
    raise ValueError(
        f"A espécie '{x_species}' possui apenas um valor ({x_min}); "
        "não é possível construir bins no eixo X."
    )

if y_max == y_min:
    raise ValueError(
        f"A espécie '{y_species}' possui apenas um valor ({y_min}); "
        "não é possível construir bins no eixo Y."
    )

matriz_local, points_local = calc_matriz_cont_any_2_species(
    filepath,
    col_X,
    col_Y,
    x_max,
    x_min,
    y_max,
    y_min,
)

matriz_final = np.zeros((dim_matriz, dim_matriz)) if rank == 0 else None
comm.Reduce(matriz_local, matriz_final, op=MPI.SUM, root=0)
points_total = comm.reduce(points_local, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Linhas numéricas utilizadas: {points_total}")
    print(f"{x_species}: min={x_min} | max={x_max}")
    print(f"{y_species}: min={y_min} | max={y_max}")

    np.savez_compressed(
        matrix_path,
        X_max=x_max,
        X_min=x_min,
        Y_max=y_max,
        Y_min=y_min,
        points_total_cont=points_total,
        matriz_contagem=matriz_final,
    )
    print("Matriz de contagem salva em:", matrix_path)

    matriz_prob = calc_matriz_prob(matriz_final, points_total)
    vetor_z_masked = calc_matriz_inv_log(matriz_prob)

    draw_surface_3d_matplotlib(
        vetor_z_masked,
        x_max,
        x_min,
        y_max,
        y_min,
        dim_matriz,
        x_species,
        y_species,
        figure_path,
    )

    print("Gráfico salvo em:", figure_path)

    end = MPI.Wtime()
    print(f"Superfície gerada com sucesso em {end - start:.2f} segundos.")
    print("==============================================")

