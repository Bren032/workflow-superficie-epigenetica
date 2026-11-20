#!/usr/bin/env python3
"""Etapa 1 - Calcula os pontos estacionários (condições iniciais) a partir do modelo SBML."""

import argparse
import sys
from pathlib import Path

import yaml
import libsbml
import sympy as sp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from itertools import product


# =========================================================
# LEITURA DO ARQUIVO DE CONFIGURAÇÃO
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


parser = argparse.ArgumentParser(
    description=(
        "Calcula os pontos iniciais/estacionários a partir do modelo SBML "
        "e das configurações do arquivo YAML."
    )
)
parser.add_argument("config", help="Arquivo YAML de configuração.")
parser.add_argument(
    "--model",
    help="Sobrescreve model.file.",
)
parser.add_argument(
    "--output",
    help="Sobrescreve stationary_points.output.",
)
args = parser.parse_args()

config_path = Path(args.config).expanduser().resolve()

if not config_path.is_file():
    raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

if not isinstance(config, dict):
    fail_config("o conteúdo principal do YAML deve ser um mapeamento/chave-valor.")

config_dir = config_path.parent

model_cfg = config.get("model")
if not isinstance(model_cfg, dict):
    fail_config("seção obrigatória 'model' ausente ou inválida.")

if not model_cfg.get("file"):
    fail_config("informe 'model.file'.")

if args.model:
    sbml_path = resolve_runtime_path(args.model)
else:
    sbml_path = resolve_config_path(model_cfg["file"], config_dir)

if not sbml_path.is_file():
    raise FileNotFoundError(f"Arquivo SBML não encontrado: {sbml_path}")

sbml_path = str(sbml_path)

stationary_cfg = config.get("stationary_points")
if not isinstance(stationary_cfg, dict):
    fail_config("seção obrigatória 'stationary_points' ausente ou inválida.")

fixed_species_raw = stationary_cfg.get("fixed_species")
if not isinstance(fixed_species_raw, list) or len(fixed_species_raw) == 0:
    fail_config("'stationary_points.fixed_species' deve ser uma lista não vazia.")

fixed_species_names = [str(s).strip() for s in fixed_species_raw if str(s).strip()]

if len(fixed_species_names) == 0:
    fail_config("nenhuma espécie fixa válida foi informada.")

if len(set(fixed_species_names)) != len(fixed_species_names):
    fail_config("existem espécies repetidas em 'stationary_points.fixed_species'.")

grid_cfg = stationary_cfg.get("grid")
if not isinstance(grid_cfg, dict):
    fail_config("seção 'stationary_points.grid' ausente ou inválida.")

for required_key in ("min", "max", "points"):
    if required_key not in grid_cfg:
        fail_config(f"informe 'stationary_points.grid.{required_key}'.")

try:
    grid_min = float(grid_cfg["min"])
    grid_max = float(grid_cfg["max"])
    grid_points = int(grid_cfg["points"])
except (TypeError, ValueError) as exc:
    fail_config(f"valores inválidos na grade: {exc}")

if not np.isfinite(grid_min) or not np.isfinite(grid_max):
    fail_config("grid.min e grid.max devem ser números finitos.")

if grid_max < grid_min:
    fail_config("grid.max deve ser maior ou igual a grid.min.")

if grid_points < 1:
    fail_config("grid.points deve ser um inteiro maior ou igual a 1.")

if not stationary_cfg.get("output"):
    fail_config("informe 'stationary_points.output'.")

if args.output:
    output_path = resolve_runtime_path(args.output)
else:
    output_path = resolve_config_path(stationary_cfg["output"], config_dir)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_file = str(output_path)

ADD_DIAGNOSTICS = bool(stationary_cfg.get("add_diagnostics", False))

# Grade de valores para cada espécie fixa
grid = {
    s: np.linspace(grid_min, grid_max, grid_points)
    for s in fixed_species_names
}


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def sanitize(name):
    """Nome interno seguro para o SymPy (não usar nos cabeçalhos do TSV)."""
    return (
        str(name)
        .replace(".", "")
        .replace("-", "")
        .replace(" ", "_")
        .replace(":", "_")
    )


def ast_type_is(t, *names, numeric=()):
    """Testa o tipo de um nó AST do libSBML (compatível entre versões)."""
    for name in names:
        if hasattr(libsbml, name):
            if t == getattr(libsbml, name):
                return True

    if t in numeric:
        return True

    return False


def ast_type_name(t):
    """Nome textual do tipo AST, para mensagens de erro."""
    for name in dir(libsbml):
        if name.startswith("AST_"):
            try:
                if getattr(libsbml, name) == t:
                    return name
            except Exception:
                pass
    return "UNKNOWN"


def ast_formula(ast):
    """Fórmula legível do nó AST."""
    try:
        return libsbml.formulaToL3String(ast)
    except Exception:
        return "<formula indisponível>"


def get_number_value(ast):
    """Extrai o valor de um nó AST numérico."""
    try:
        return ast.getValue()
    except Exception:
        pass

    try:
        return ast.getInteger()
    except Exception:
        pass

    try:
        return float(ast.getReal())
    except Exception:
        pass

    raise ValueError(f"Não foi possível extrair valor numérico de AST: {ast_formula(ast)}")


def ast_to_sympy(ast, symbols, functions):
    """Converte uma árvore AST do SBML em expressão SymPy."""

    if ast is None:
        raise ValueError("AST vazio recebido em ast_to_sympy")

    t = ast.getType()
    nchild = ast.getNumChildren()

    # -------------------------
    # Números
    # -------------------------

    if hasattr(ast, "isNumber") and ast.isNumber():
        val = get_number_value(ast)
        if float(val).is_integer():
            return sp.Integer(int(val))
        return sp.Float(val)

    if ast_type_is(t, "AST_INTEGER"):
        return sp.Integer(int(get_number_value(ast)))

    if ast_type_is(t, "AST_REAL", "AST_REAL_E"):
        return sp.Float(get_number_value(ast))

    if ast_type_is(t, "AST_RATIONAL"):
        if nchild == 2:
            num = ast_to_sympy(ast.getChild(0), symbols, functions)
            den = ast_to_sympy(ast.getChild(1), symbols, functions)
            return num / den
        return sp.Float(get_number_value(ast))

    # -------------------------
    # Constantes matemáticas
    # -------------------------

    if ast_type_is(t, "AST_CONSTANT_PI"):
        return sp.pi

    if ast_type_is(t, "AST_CONSTANT_E"):
        return sp.E

    if ast_type_is(t, "AST_CONSTANT_TRUE"):
        return sp.true

    if ast_type_is(t, "AST_CONSTANT_FALSE"):
        return sp.false

    # -------------------------
    # Nomes / variáveis
    # -------------------------

    if ast_type_is(t, "AST_NAME"):
        name = sanitize(ast.getName())
        if name not in symbols:
            symbols[name] = sp.Symbol(name)
        return symbols[name]

    if ast_type_is(t, "AST_NAME_TIME"):
        name = "time"
        if name not in symbols:
            symbols[name] = sp.Symbol(name)
        return symbols[name]

    # -------------------------
    # Operadores aritméticos
    # -------------------------

    if ast_type_is(t, "AST_PLUS"):
        return sum(
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        )

    if ast_type_is(t, "AST_MINUS"):
        if nchild == 1:
            return -ast_to_sympy(ast.getChild(0), symbols, functions)

        if nchild == 2:
            return (
                ast_to_sympy(ast.getChild(0), symbols, functions)
                - ast_to_sympy(ast.getChild(1), symbols, functions)
            )

        raise ValueError(f"AST_MINUS com {nchild} filhos: {ast_formula(ast)}")

    if ast_type_is(t, "AST_TIMES"):
        out = sp.Integer(1)
        for i in range(nchild):
            out *= ast_to_sympy(ast.getChild(i), symbols, functions)
        return out

    if ast_type_is(t, "AST_DIVIDE"):
        if nchild != 2:
            raise ValueError(f"AST_DIVIDE com {nchild} filhos: {ast_formula(ast)}")

        return (
            ast_to_sympy(ast.getChild(0), symbols, functions)
            / ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_POWER", "AST_FUNCTION_POWER", numeric=(296,)):
        if nchild != 2:
            raise ValueError(f"POWER com {nchild} filhos: {ast_formula(ast)}")

        base = ast_to_sympy(ast.getChild(0), symbols, functions)
        expo = ast_to_sympy(ast.getChild(1), symbols, functions)

        return base ** expo

    # -------------------------
    # Funções matemáticas nativas
    # -------------------------

    if ast_type_is(t, "AST_FUNCTION_EXP"):
        return sp.exp(ast_to_sympy(ast.getChild(0), symbols, functions))

    if ast_type_is(t, "AST_FUNCTION_LN"):
        return sp.log(ast_to_sympy(ast.getChild(0), symbols, functions))

    if ast_type_is(t, "AST_FUNCTION_LOG"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]

        if len(args) == 1:
            return sp.log(args[0])

        if len(args) == 2:
            return sp.log(args[1]) / sp.log(args[0])

        raise ValueError(f"LOG com {nchild} filhos: {ast_formula(ast)}")

    if ast_type_is(t, "AST_FUNCTION_ROOT"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]

        if len(args) == 1:
            return sp.sqrt(args[0])

        if len(args) == 2:
            degree, value = args
            return value ** (sp.Integer(1) / degree)

        raise ValueError(f"ROOT com {nchild} filhos: {ast_formula(ast)}")

    if ast_type_is(t, "AST_FUNCTION_ABS"):
        return sp.Abs(ast_to_sympy(ast.getChild(0), symbols, functions))

    if ast_type_is(t, "AST_FUNCTION_FLOOR"):
        return sp.floor(ast_to_sympy(ast.getChild(0), symbols, functions))

    if ast_type_is(t, "AST_FUNCTION_CEILING"):
        return sp.ceiling(ast_to_sympy(ast.getChild(0), symbols, functions))

    if ast_type_is(t, "AST_FUNCTION_MAX"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]
        return sp.Max(*args)

    if ast_type_is(t, "AST_FUNCTION_MIN"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]
        return sp.Min(*args)

    # -------------------------
    # Relacionais
    # -------------------------

    if ast_type_is(t, "AST_RELATIONAL_LT"):
        return sp.StrictLessThan(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_RELATIONAL_LEQ"):
        return sp.LessThan(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_RELATIONAL_GT"):
        return sp.StrictGreaterThan(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_RELATIONAL_GEQ"):
        return sp.GreaterThan(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_RELATIONAL_EQ"):
        return sp.Eq(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    if ast_type_is(t, "AST_RELATIONAL_NEQ"):
        return sp.Ne(
            ast_to_sympy(ast.getChild(0), symbols, functions),
            ast_to_sympy(ast.getChild(1), symbols, functions)
        )

    # -------------------------
    # Lógicos
    # -------------------------

    if ast_type_is(t, "AST_LOGICAL_AND"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]
        return sp.And(*args)

    if ast_type_is(t, "AST_LOGICAL_OR"):
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]
        return sp.Or(*args)

    if ast_type_is(t, "AST_LOGICAL_NOT"):
        return sp.Not(ast_to_sympy(ast.getChild(0), symbols, functions))

    # -------------------------
    # Piecewise
    # -------------------------

    if ast_type_is(t, "AST_FUNCTION_PIECEWISE"):
        pieces = []

        i = 0
        while i + 1 < nchild:
            value = ast_to_sympy(ast.getChild(i), symbols, functions)
            condition = ast_to_sympy(ast.getChild(i + 1), symbols, functions)
            pieces.append((value, condition))
            i += 2

        if i < nchild:
            otherwise = ast_to_sympy(ast.getChild(i), symbols, functions)
            pieces.append((otherwise, True))

        return sp.Piecewise(*pieces)

    # -------------------------
    # Funções genéricas / definidas pelo usuário
    # -------------------------

    if ast_type_is(t, "AST_FUNCTION"):
        fname = ast.getName()
        args = [
            ast_to_sympy(ast.getChild(i), symbols, functions)
            for i in range(nchild)
        ]

        if fname == "exp":
            return sp.exp(args[0])

        if fname in ("log", "ln"):
            return sp.log(args[0])

        if fname == "sqrt":
            return sp.sqrt(args[0])

        if fname == "abs":
            return sp.Abs(args[0])

        if fname in ("pow", "power"):
            return args[0] ** args[1]

        if fname == "max":
            return sp.Max(*args)

        if fname == "min":
            return sp.Min(*args)

        # FunctionDefinition do SBML
        if fname in functions:
            f_args, f_body = functions[fname]
            subs = dict(zip(f_args, args))
            return f_body.subs(subs)

        raise NotImplementedError(f"Função SBML não suportada: {fname}")

    # -------------------------
    # Lambda
    # -------------------------

    if ast_type_is(t, "AST_LAMBDA"):
        if nchild < 1:
            raise ValueError("Lambda SBML sem corpo")

        return ast_to_sympy(ast.getChild(nchild - 1), symbols, functions)

    # -------------------------
    # Erro informativo
    # -------------------------

    raise NotImplementedError(
        f"AST não suportado: "
        f"type={t}, "
        f"name={ast_type_name(t)}, "
        f"node_name={ast.getName()}, "
        f"formula={ast_formula(ast)}"
    )


# =========================================================
# LEITURA DO SBML
# =========================================================

doc = libsbml.readSBML(sbml_path)

if doc is None:
    raise RuntimeError(f"Não foi possível ler o arquivo SBML: {sbml_path}")

if doc.getNumErrors() > 0:
    print("Avisos/erros reportados pelo libSBML:")
    doc.printErrors()

model = doc.getModel()

if model is None:
    raise RuntimeError("Erro ao carregar modelo SBML: model == None")

symbols = {}
param_syms = []
param_vals = []


# =========================================================
# FUNCTION DEFINITIONS DO SBML
# =========================================================

functions = {}

for fd in model.getListOfFunctionDefinitions():
    fname = fd.getId()
    math = fd.getMath()

    if math is None:
        continue

    args = []

    if ast_type_is(math.getType(), "AST_LAMBDA"):
        for i in range(math.getNumChildren() - 1):
            arg_name = sanitize(math.getChild(i).getName())
            arg_sym = sp.Symbol(arg_name)
            symbols[arg_name] = arg_sym
            args.append(arg_sym)

        body_ast = math.getChild(math.getNumChildren() - 1)
        body = ast_to_sympy(body_ast, symbols, functions)
        functions[fname] = (args, body)

    else:
        for i in range(math.getNumChildren() - 1):
            arg_name = sanitize(math.getChild(i).getName())
            arg_sym = sp.Symbol(arg_name)
            symbols[arg_name] = arg_sym
            args.append(arg_sym)

        body_ast = math.getChild(math.getNumChildren() - 1)
        body = ast_to_sympy(body_ast, symbols, functions)
        functions[fname] = (args, body)


# =========================================================
# COMPARTIMENTOS
# =========================================================

for c in model.getListOfCompartments():
    cid = sanitize(c.getId())
    s = sp.Symbol(cid)
    symbols[cid] = s

    if c.isSetSize():
        value = c.getSize()
    elif c.isSetVolume():
        value = c.getVolume()
    else:
        value = 1.0

    param_syms.append(s)
    param_vals.append(float(value))


# =========================================================
# PARÂMETROS GLOBAIS
# =========================================================

for p in model.getListOfParameters():
    pid = sanitize(p.getId())
    s = sp.Symbol(pid)
    symbols[pid] = s

    if p.isSetValue():
        value = p.getValue()
    else:
        value = 0.0

    param_syms.append(s)
    param_vals.append(float(value))


# =========================================================
# ESPÉCIES
# =========================================================

species = []

# species_ids: ID do SBML | species_internal_ids: nome SymPy | species_output_names: nome COPASI (TSV)
species_ids = []
species_internal_ids = []
species_output_names = []
y0 = []

for spc in model.getListOfSpecies():
    raw_sid = str(spc.getId())
    internal_sid = sanitize(raw_sid)
    sym = sp.Symbol(internal_sid)

    if spc.isSetName() and str(spc.getName()).strip():
        output_name = str(spc.getName()).strip()
    else:
        output_name = raw_sid

    symbols[internal_sid] = sym
    species.append(sym)
    species_ids.append(raw_sid)
    species_internal_ids.append(internal_sid)
    species_output_names.append(output_name)

    if spc.isSetInitialConcentration():
        val = spc.getInitialConcentration()
    elif spc.isSetInitialAmount():
        val = spc.getInitialAmount()
    else:
        val = 1.0

    y0.append(max(float(val), 1e-8))

y0 = np.array(y0, dtype=float)

if len(species) == 0:
    raise RuntimeError("O modelo SBML não possui espécies.")

if len(set(species_output_names)) != len(species_output_names):
    duplicates = sorted({
        name for name in species_output_names
        if species_output_names.count(name) > 1
    })
    raise ValueError(
        "Existem espécies com nomes COPASI duplicados: "
        + ", ".join(duplicates)
        + ". O TSV não pode ter cabeçalhos duplicados; corrija os nomes no modelo."
    )


# =========================================================
# VALIDAÇÃO E RESOLUÇÃO DAS ESPÉCIES FIXAS
# =========================================================

def resolve_species_index(requested_name):
    """Encontra a espécie pelo nome COPASI, ID SBML ou nome sanitizado."""
    requested_name = str(requested_name).strip()

    exact_matches = [
        i for i, (sid, output_name) in enumerate(
            zip(species_ids, species_output_names)
        )
        if requested_name == sid or requested_name == output_name
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise ValueError(
            f"O nome de espécie '{requested_name}' é ambíguo no modelo."
        )

    requested_internal = sanitize(requested_name)
    fallback_matches = [
        i for i, internal_sid in enumerate(species_internal_ids)
        if requested_internal == internal_sid
    ]

    if len(fallback_matches) == 1:
        return fallback_matches[0]

    if len(fallback_matches) > 1:
        raise ValueError(
            f"O nome sanitizado da espécie '{requested_name}' é ambíguo."
        )

    available = [
        f"{name} [SBML id: {sid}]"
        for name, sid in zip(species_output_names, species_ids)
    ]
    raise ValueError(
        f"Espécie fixa não encontrada no modelo: {requested_name}"
        + "\nEspécies disponíveis: "
        + ", ".join(available)
    )


idx_fixed = [resolve_species_index(name) for name in fixed_species_names]

if len(set(idx_fixed)) != len(idx_fixed):
    raise ValueError(
        "A mesma espécie fixa foi informada mais de uma vez, "
        "possivelmente usando simultaneamente o nome COPASI e o ID SBML."
    )

fixed_syms = [species[i] for i in idx_fixed]
idx_free = [i for i in range(len(species)) if i not in idx_fixed]
free_syms = [species[i] for i in idx_free]

if len(free_syms) == 0:
    raise ValueError(
        "Você fixou todas as espécies do modelo. "
        "Não há espécies livres para o solver ajustar. "
        "Para esse tipo de código, fixe apenas parte das espécies."
    )


# =========================================================
# CONSTRUÇÃO DAS EDOs E MATRIZ ESTEQUIOMÉTRICA
# =========================================================

odes = [sp.Integer(0) for _ in species]
idx_map = {s: i for i, s in enumerate(species)}

num_species = len(species)
num_reactions = model.getNumReactions()

S_matrix = np.zeros((num_species, num_reactions), dtype=float)

for r_idx, r in enumerate(model.getListOfReactions()):

    kl = r.getKineticLaw()

    if kl is None or kl.getMath() is None:
        raise RuntimeError(f"Reação {r.getId()} não possui kineticLaw/math.")

    rate = ast_to_sympy(kl.getMath(), symbols, functions)

    for lp in kl.getListOfParameters():
        lp_id = sanitize(lp.getId())
        lp_sym = sp.Symbol(lp_id)
        symbols[lp_id] = lp_sym

        if lp.isSetValue():
            rate = rate.subs(lp_sym, float(lp.getValue()))

    if hasattr(kl, "getListOfLocalParameters"):
        for lp in kl.getListOfLocalParameters():
            lp_id = sanitize(lp.getId())
            lp_sym = sp.Symbol(lp_id)
            symbols[lp_id] = lp_sym

            if lp.isSetValue():
                rate = rate.subs(lp_sym, float(lp.getValue()))

    for re in r.getListOfReactants():
        sid = sanitize(re.getSpecies())
        sp_sym = symbols[sid]
        sp_idx = idx_map[sp_sym]

        stoich = re.getStoichiometry()
        if stoich is None:
            stoich = 1.0

        stoich = float(stoich)

        odes[sp_idx] -= stoich * rate
        S_matrix[sp_idx, r_idx] -= stoich

    for pr in r.getListOfProducts():
        sid = sanitize(pr.getSpecies())
        sp_sym = symbols[sid]
        sp_idx = idx_map[sp_sym]

        stoich = pr.getStoichiometry()
        if stoich is None:
            stoich = 1.0

        stoich = float(stoich)

        odes[sp_idx] += stoich * rate
        S_matrix[sp_idx, r_idx] += stoich


# =========================================================
# CÁLCULO DAS LEIS DE CONSERVAÇÃO
# =========================================================

# Espaço nulo à esquerda da matriz estequiométrica (c^T S = 0)
S_sympy = sp.Matrix(S_matrix)
left_null_space = S_sympy.T.nullspace()

if len(left_null_space) > 0:
    conservation_vectors = np.array(
        [[float(val) for val in vec] for vec in left_null_space],
        dtype=float
    )
    conservation_totals = conservation_vectors.dot(y0)
else:
    conservation_vectors = np.empty((0, num_species), dtype=float)
    conservation_totals = np.array([], dtype=float)


# =========================================================
# FUNÇÃO NUMÉRICA DAS EDOs
# =========================================================

f_num = sp.lambdify(
    [species, param_syms],
    odes,
    modules="numpy"
)

param_vals = np.array(param_vals, dtype=float)


def evaluate_odes(x):
    val = np.array(f_num(x, param_vals), dtype=float).reshape(-1)

    if not np.all(np.isfinite(val)):
        val = np.where(np.isfinite(val), val, 1e12)

    return val


def field_free(x_free, fixed_vals):
    """Resíduos do least_squares: ajusta só as espécies livres."""

    x = np.zeros(len(species), dtype=float)

    x[idx_free] = np.asarray(x_free, dtype=float)
    x[idx_fixed] = np.asarray(fixed_vals, dtype=float)

    ode_all = evaluate_odes(x)

    ode_res = ode_all[idx_free]

    if len(conservation_vectors) > 0:
        cons_res = conservation_vectors.dot(x) - conservation_totals

        return np.concatenate([ode_res, cons_res * 1e3])

    return ode_res


def full_state_from_solution(sol_x, fixed_vals):
    x = np.zeros(len(species), dtype=float)
    x[idx_free] = np.asarray(sol_x, dtype=float)
    x[idx_fixed] = np.asarray(fixed_vals, dtype=float)
    return x


# =========================================================
# VARREDURA DA GRADE
# =========================================================

rows = []
last = None

total_points = grid_points ** len(fixed_species_names)
counter = 0

print("==============================================")
print("Modelo:", sbml_path)
print(
    "Espécies do modelo:",
    ", ".join(
        f"{name} [id={sid}]"
        for name, sid in zip(species_output_names, species_ids)
    )
)
print(
    "Espécies fixas:",
    ", ".join(species_output_names[i] for i in idx_fixed)
)
print(
    "Espécies livres:",
    ", ".join(species_output_names[i] for i in idx_free)
)
print("Pontos na grade:", total_points)
print("Intervalo da grade:", grid_min, "até", grid_max)
print("==============================================")

for fixed_vals in product(*(grid[s] for s in fixed_species_names)):

    counter += 1

    if last is not None:
        x0 = last.copy()
    else:
        base = y0[idx_free].copy()

        if np.all(base <= 1e-7):
            initial_value = max((grid_min + grid_max) / 2.0, 1e-3)
            x0 = np.full(len(idx_free), initial_value, dtype=float)
        else:
            x0 = np.maximum(base, 1e-6)

    sol = least_squares(
        field_free,
        x0,
        args=(fixed_vals,),
        bounds=(0, np.inf),
        max_nfev=4000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10
    )

    x_full = full_state_from_solution(sol.x, fixed_vals)

    row = {
        species_output_names[i]: float(x_full[i])
        for i in range(len(species))
    }

    if ADD_DIAGNOSTICS:
        ode_all = evaluate_odes(x_full)
        free_res = ode_all[idx_free]

        row["solver_success"] = bool(sol.success)
        row["solver_status"] = int(sol.status)
        row["solver_cost"] = float(sol.cost)
        row["solver_nfev"] = int(sol.nfev)
        row["max_abs_free_ode_residual"] = float(np.max(np.abs(free_res)))

        if len(conservation_vectors) > 0:
            cons_res = conservation_vectors.dot(x_full) - conservation_totals
            row["max_abs_conservation_residual"] = float(np.max(np.abs(cons_res)))
        else:
            row["max_abs_conservation_residual"] = 0.0

    rows.append(row)

    last = sol.x.copy()

    if counter % 100 == 0 or counter == total_points:
        print(f"Processado {counter}/{total_points} pontos")


# =========================================================
# SALVAR RESULTADO
# =========================================================

df = pd.DataFrame(rows)

main_cols = species_output_names.copy()

if ADD_DIAGNOSTICS:
    diag_cols = [c for c in df.columns if c not in main_cols]
    df = df[main_cols + diag_cols]
else:
    df = df[main_cols]

df.to_csv(output_file, sep="\t", index=False)

print("==============================================")
print("Superfície epigenética construída com sucesso")
print("Arquivo salvo em:", output_file)
print("Número de linhas:", len(df))
print("Número de colunas:", len(df.columns))
print("==============================================")

