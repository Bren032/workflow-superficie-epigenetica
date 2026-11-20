#!/usr/bin/env python3
"""Etapa 2 - Simula cada ponto inicial no COPASI e gera o landscape (sem cabeçalho)."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

try:
    import basico
except ImportError as exc:
    raise ImportError(
        "BasiCO não está instalado. Instale com:\n"
        "    pip install copasi-basico\n"
        "e execute novamente."
    ) from exc


# =========================================================
# UTILITÁRIOS DE CONFIGURAÇÃO
# =========================================================


def fail_config(message):
    raise ValueError(f"Erro no arquivo de configuração: {message}")


def resolve_config_path(value, config_dir):
    """Resolve caminhos do YAML relativos à pasta do config.yaml."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def resolve_runtime_path(value):
    """Resolve caminhos da linha de comando relativos ao diretório atual."""
    return Path(str(value)).expanduser().resolve()


def require_mapping(config, key):
    value = config.get(key)
    if not isinstance(value, dict):
        fail_config(f"seção obrigatória '{key}' ausente ou inválida.")
    return value


# =========================================================
# ARGUMENTOS
# =========================================================

parser = argparse.ArgumentParser(
    description=(
        "Executa as simulações para a paisagem epigenética usando um modelo "
        "SBML + config.yaml. O BasiCO prepara automaticamente um CPS temporário "
        "com Time-Course e Report; o CopasiSE executa as simulações."
    )
)
parser.add_argument("config", help="Arquivo YAML de configuração.")
parser.add_argument(
    "--model",
    help="Sobrescreve model.file.",
)
parser.add_argument(
    "--points",
    help="Arquivo de pontos (chunk) a simular.",
)
parser.add_argument(
    "--landscape",
    help="Sobrescreve output.landscape.",
)
parser.add_argument(
    "--keep-prepared-cps",
    action="store_true",
    help="Mantém uma cópia do CPS preparado (depuração).",
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
# MODELO SBML
# =========================================================

model_cfg = require_mapping(config, "model")
if not model_cfg.get("file"):
    fail_config("informe 'model.file'.")

if args.model:
    sbml_path = resolve_runtime_path(args.model)
else:
    sbml_path = resolve_config_path(model_cfg["file"], config_dir)

if not sbml_path.is_file():
    raise FileNotFoundError(f"Arquivo SBML não encontrado: {sbml_path}")


# =========================================================
# ARQUIVO DE PONTOS
# =========================================================

stationary_cfg = require_mapping(config, "stationary_points")

if args.points:
    points_path = resolve_runtime_path(args.points)
else:
    if not stationary_cfg.get("output"):
        fail_config("informe 'stationary_points.output'.")
    points_path = resolve_config_path(stationary_cfg["output"], config_dir)

if not points_path.is_file():
    raise FileNotFoundError(f"Arquivo de pontos iniciais não encontrado: {points_path}")


# =========================================================
# COPASI / CopasiSE
# =========================================================

copasi_cfg = require_mapping(config, "copasi")

copasi_executable = copasi_cfg.get("executable")
install_path = copasi_cfg.get("install_path")

if copasi_executable:
    copasi_se = resolve_config_path(copasi_executable, config_dir)
elif install_path:
    install_dir = resolve_config_path(install_path, config_dir)
    copasi_se = install_dir / "bin" / "CopasiSE"
else:
    found = shutil.which("CopasiSE")
    if found is None:
        fail_config(
            "informe 'copasi.install_path' ou 'copasi.executable', "
            "ou deixe CopasiSE disponível no PATH."
        )
    copasi_se = Path(found).resolve()

if not copasi_se.is_file():
    raise FileNotFoundError(f"Executável CopasiSE não encontrado: {copasi_se}")


# =========================================================
# CONFIGURAÇÃO DA SIMULAÇÃO
# =========================================================

simulation_cfg = require_mapping(config, "simulation")

try:
    duration = float(simulation_cfg.get("duration", 1440))
    intervals = int(simulation_cfg.get("intervals", 1440))
    num_sims_per_point = int(simulation_cfg.get("simulations_per_point", 1))
    output_start_time = float(simulation_cfg.get("output_start_time", 0.0))
except (TypeError, ValueError) as exc:
    fail_config(f"valores inválidos em 'simulation': {exc}")

if duration <= 0:
    fail_config("'simulation.duration' deve ser maior que zero.")
if intervals < 1:
    fail_config("'simulation.intervals' deve ser >= 1.")
if num_sims_per_point < 1:
    fail_config("'simulation.simulations_per_point' deve ser >= 1.")
if output_start_time < 0 or output_start_time > duration:
    fail_config("'simulation.output_start_time' deve estar entre 0 e duration.")

method_raw = str(simulation_cfg.get("method", "deterministic")).strip()

METHOD_ALIASES = {
    "deterministic": "Deterministic (LSODA)",
    "lsoda": "Deterministic (LSODA)",
    "deterministic_lsoda": "Deterministic (LSODA)",
    "radau5": "Deterministic (RADAU5)",
    "deterministic_radau5": "Deterministic (RADAU5)",
    "tau_leap": "Stochastic (τ-Leap)",
    "stochastic_tau_leap": "Stochastic (τ-Leap)",
    "τ-leap": "Stochastic (τ-Leap)",
    "direct": "Stochastic (Direct method)",
    "stochastic_direct": "Stochastic (Direct method)",
    "gibson_bruck": "Stochastic (Gibson + Bruck)",
    "stochastic_gibson_bruck": "Stochastic (Gibson + Bruck)",
    "adaptive_tau_leap": "Stochastic (Adaptive SSA/τ-Leap)",
    "stochastic_adaptive_tau_leap": "Stochastic (Adaptive SSA/τ-Leap)",
}

method_name = METHOD_ALIASES.get(method_raw.lower(), method_raw)

method_parameters = simulation_cfg.get("method_parameters", {})
if method_parameters is None:
    method_parameters = {}
if not isinstance(method_parameters, dict):
    fail_config("'simulation.method_parameters' deve ser um mapeamento/chave-valor.")


# =========================================================
# ORDEM DO REPORT
# =========================================================

report_cfg = require_mapping(config, "report")
report_species_raw = report_cfg.get("species")

if not isinstance(report_species_raw, list) or len(report_species_raw) == 0:
    fail_config("'report.species' deve ser uma lista não vazia e ordenada.")

report_species_requested = [str(x).strip() for x in report_species_raw if str(x).strip()]

if len(report_species_requested) != len(report_species_raw):
    fail_config("'report.species' contém um nome vazio.")

if len(set(report_species_requested)) != len(report_species_requested):
    fail_config("existem espécies repetidas em 'report.species'.")

if any(name.lower() == "time" for name in report_species_requested):
    fail_config(
        "não inclua 'Time' em 'report.species': o programa adiciona Time "
        "automaticamente como a primeira coluna."
    )

try:
    report_precision = int(report_cfg.get("precision", 12))
except (TypeError, ValueError):
    fail_config("'report.precision' deve ser um número inteiro.")

if report_precision < 1:
    fail_config("'report.precision' deve ser >= 1.")


# =========================================================
# SAÍDA LANDSCAPE
# =========================================================

output_cfg = require_mapping(config, "output")

if args.landscape:
    landscape_path = resolve_runtime_path(args.landscape)
else:
    if not output_cfg.get("landscape"):
        fail_config("informe 'output.landscape'.")
    landscape_path = resolve_config_path(output_cfg["landscape"], config_dir)

landscape_path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# FUNÇÕES BasiCO
# =========================================================


def resolve_copasi_species(requested_name, species_frame):
    """Resolve um nome (COPASI, sbml_id ou display_name) para o nome COPASI."""
    requested_name = str(requested_name).strip()

    index_names = [str(x) for x in species_frame.index.tolist()]

    exact_index = [name for name in index_names if name == requested_name]
    if len(exact_index) == 1:
        return exact_index[0]

    if "sbml_id" in species_frame.columns:
        matches = []
        for idx, value in species_frame["sbml_id"].items():
            if str(value) == requested_name:
                matches.append(str(idx))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"O SBML id '{requested_name}' corresponde a mais de uma espécie no COPASI."
            )

    if "display_name" in species_frame.columns:
        matches = []
        for idx, value in species_frame["display_name"].items():
            if str(value) == requested_name:
                matches.append(str(idx))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"O display_name '{requested_name}' corresponde a mais de uma espécie."
            )

    available = []
    for idx, row in species_frame.iterrows():
        sbml_id = row.get("sbml_id", "")
        available.append(f"{idx} [SBML id: {sbml_id}]")

    raise ValueError(
        f"Espécie '{requested_name}' não encontrada pelo BasiCO/COPASI.\n"
        "Espécies disponíveis:\n  - " + "\n  - ".join(available)
    )


def prepare_copasi_model(prepared_cps, default_report_file):
    """Carrega o SBML, configura Time-Course e Report e salva um CPS temporário."""
    print("==============================================")
    print("Preparando modelo COPASI automaticamente com BasiCO")
    print("SBML:", sbml_path)
    print("==============================================")

    model = basico.load_model(str(sbml_path))
    if model is None:
        raise RuntimeError(f"BasiCO não conseguiu carregar o SBML: {sbml_path}")

    species_frame = basico.get_species(model=model)
    if species_frame is None or len(species_frame) == 0:
        raise RuntimeError("O modelo carregado pelo BasiCO não possui espécies.")

    resolved_report_species = [
        resolve_copasi_species(name, species_frame)
        for name in report_species_requested
    ]

    valid_methods = list(basico.get_valid_methods(basico.T.TIME_COURSE, model=model))
    if method_name not in valid_methods:
        raise ValueError(
            f"Método '{method_name}' não é válido para Time-Course nesta instalação.\n"
            "Métodos disponíveis:\n  - " + "\n  - ".join(map(str, valid_methods))
        )

    # O método deve ser definido antes dos seus parâmetros
    basico.set_task_settings(
        basico.T.TIME_COURSE,
        settings={"method": {"name": method_name}},
        model=model,
    )

    step_size = duration / intervals
    problem_settings = {
        "AutomaticStepSize": False,
        "StepNumber": intervals,
        "StepSize": step_size,
        "Duration": duration,
        "TimeSeriesRequested": True,
        "OutputStartTime": output_start_time,
        "Start in Steady State": False,
    }

    basico.set_task_settings(
        basico.T.TIME_COURSE,
        settings={"problem": problem_settings},
        model=model,
    )

    # Parâmetros opcionais do método (simulation.method_parameters)
    if method_parameters:
        current_method = dict(
            basico.get_task_settings(basico.T.TIME_COURSE, model=model)["method"]
        )

        unknown = [key for key in method_parameters if key not in current_method]
        if unknown:
            raise ValueError(
                "Parâmetro(s) não reconhecido(s) para o método "
                f"'{method_name}': {', '.join(unknown)}.\n"
                "Parâmetros aceitos nesta instalação: "
                + ", ".join(current_method.keys())
            )

        updated_method = {"name": method_name}
        updated_method.update(method_parameters)

        basico.set_task_settings(
            basico.T.TIME_COURSE,
            settings={"method": updated_method},
            model=model,
        )

    # Report: Time + concentração [Espécie] na ordem de report.species
    report_name = "Landscape Time-Course"

    try:
        basico.remove_report(report_name, model=model)
    except Exception:
        pass

    report_table = ["Time"] + [f"[{name}]" for name in resolved_report_species]

    basico.add_report(
        report_name,
        task=basico.T.TIME_COURSE,
        table=report_table,
        model=model,
    )

    basico.set_report_dict(
        report_name,
        separator="\t",
        precision=report_precision,
        print_headers=True,
        model=model,
    )

    basico.assign_report(
        report_name,
        task=basico.T.TIME_COURSE,
        filename=str(default_report_file),
        append=False,
        model=model,
    )

    basico.set_scheduled_tasks(basico.T.TIME_COURSE, model=model)

    basico.save_model(str(prepared_cps), model=model)

    final_settings = basico.get_task_settings(basico.T.TIME_COURSE, model=model)

    print("Método:", final_settings["method"].get("name"))
    print("Duração:", final_settings["problem"].get("Duration"))
    print("Intervalos:", final_settings["problem"].get("StepNumber"))
    print("Passo de saída:", final_settings["problem"].get("StepSize"))
    print("Ordem do Report:")
    print("  coluna 0 -> Time")
    for i, name in enumerate(resolved_report_species, start=1):
        print(f"  coluna {i} -> {name}")
    print("CPS temporário preparado:", prepared_cps)
    print("==============================================")

    return species_frame, resolved_report_species


# =========================================================
# PREPARAÇÃO E EXECUÇÃO
# =========================================================

with tempfile.TemporaryDirectory(prefix="landscape_copasi_") as tmp_dir_name:
    tmp_dir = Path(tmp_dir_name)
    prepared_cps = tmp_dir / "prepared_model.cps"
    default_report_file = tmp_dir / "default_report.txt"
    start_point_file = tmp_dir / "startPoint.ini"
    model_results_file = tmp_dir / "modelResults.txt"

    species_frame, resolved_report_species = prepare_copasi_model(
        prepared_cps,
        default_report_file,
    )

    if args.keep_prepared_cps:
        debug_cps = landscape_path.parent / "prepared_model_debug.cps"
        shutil.copy2(prepared_cps, debug_cps)
        print("Cópia do CPS preparado mantida em:", debug_cps)

    # Cabeçalho do arquivo de pontos -> nomes de espécies do COPASI
    with points_path.open("r", encoding="utf-8") as points_file:
        header = points_file.readline()
        if not header:
            raise ValueError(f"Arquivo de pontos vazio: {points_path}")

        point_species_requested = header.rstrip("\n\r").split("\t")
        if len(point_species_requested) == 0:
            raise ValueError("Não foi possível ler as espécies do cabeçalho dos pontos.")

        point_species_copasi = [
            resolve_copasi_species(name, species_frame)
            for name in point_species_requested
        ]

        print("Arquivo de pontos:", points_path)
        print("Landscape de saída:", landscape_path)
        print("Simulações por ponto:", num_sims_per_point)
        print("CopasiSE:", copasi_se)
        print("==============================================")

        with landscape_path.open("w", encoding="utf-8") as landscape_file:
            pointnum = 1

            for line_number, point in enumerate(points_file, start=2):
                stripped = point.rstrip("\n\r")
                if not stripped:
                    continue

                values = stripped.split("\t")

                if len(values) != len(point_species_copasi):
                    raise ValueError(
                        f"Linha {line_number} do arquivo de pontos possui "
                        f"{len(values)} valores, mas o cabeçalho possui "
                        f"{len(point_species_copasi)} espécies."
                    )

                # Condições iniciais do ponto
                with start_point_file.open("w", encoding="utf-8") as ini_file:
                    for specie, value in zip(point_species_copasi, values):
                        ini_file.write(f"[{specie}]_0 = {value}\n")

                for simnum in range(num_sims_per_point):
                    print(
                        f"Point #{pointnum:03}, "
                        f"Simulation #{simnum + 1:04}"
                    )

                    if model_results_file.exists():
                        model_results_file.unlink()

                    command = [
                        str(copasi_se),
                        "--scheduled-task",
                        "Time-Course",
                        "--reparameterize",
                        str(start_point_file),
                        "--report-file",
                        str(model_results_file),
                        str(prepared_cps),
                    ]

                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                    )

                    if process.returncode != 0:
                        print("\n================ ERRO NO COPASI ================")
                        print(f"Ponto: {pointnum}")
                        print(f"Simulação: {simnum + 1}")
                        print(f"Código de saída: {process.returncode}")
                        print("Comando:")
                        print(" ".join(command))
                        print("--- MENSAGEM DE ERRO (stderr) ---")
                        print(process.stderr)
                        print("--- SAÍDA PADRÃO (stdout) ---")
                        print(process.stdout)
                        print("================================================\n")
                        sys.exit(1)

                    if not model_results_file.is_file():
                        raise RuntimeError(
                            "CopasiSE terminou sem erro, mas não criou o arquivo "
                            f"de Report esperado: {model_results_file}"
                        )

                    with model_results_file.open("r", encoding="utf-8") as results_file:
                        # Descarta o cabeçalho: o landscape é gravado sem cabeçalho
                        header_line = results_file.readline()

                        if not header_line:
                            raise RuntimeError(
                                "O arquivo de Report do COPASI está vazio."
                            )

                        wrote_data = False
                        for result_line in results_file:
                            landscape_file.write(result_line)
                            wrote_data = True

                        if not wrote_data:
                            raise RuntimeError(
                                "O Report do COPASI contém cabeçalho, mas nenhuma "
                                "linha numérica de Time-Course."
                            )

                        landscape_file.write("\n")
                        landscape_file.flush()

                pointnum += 1

print("==============================================")
print("Simulações finalizadas com sucesso")
print("Landscape salvo em:", landscape_path)
print("O arquivo final permanece SEM cabeçalho.")
print("==============================================")

