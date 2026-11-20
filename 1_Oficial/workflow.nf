#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process ENCONTRAR_PONTOS_ESTACIONARIOS {

    publishDir "${params.workflow.outdir}", mode: 'copy'

    input:
    path script_python
    path config_yaml
    path xml_modelo
    val output_name
    val activate_cmd

    output:
    path "${output_name}", emit: pontos_tsv

    script:
    """
    ${activate_cmd}

    python3 ${script_python} ${config_yaml} \
        --model ${xml_modelo} \
        --output ${output_name}
    """
}


process DIVIDIR_PONTOS {

    input:
    path tsv_original
    val tamanho_pedaco

    output:
    path "pedaco_*.tsv", emit: pedacos_ch

    script:
    """
    head -n 1 ${tsv_original} > header.txt

    tail -n +2 ${tsv_original} | \
        split -l ${tamanho_pedaco} - pedaco_raw_

    for f in pedaco_raw_*; do
        cat header.txt "\$f" > "pedaco_\${f#pedaco_raw_}.tsv"
    done

    rm -f header.txt pedaco_raw_*
    """
}


process GERAR_LANDSCAPE {

    input:
    path tsv_pedaco
    path script_python
    path config_yaml
    path xml_modelo
    val activate_cmd
    val launcher

    output:
    path "resultado_*.tsv", emit: resultado_parcial

    script:
    """
    ${activate_cmd}

    chunk_base=\$(basename "${tsv_pedaco}" .tsv)
    partial_file="resultado_\${chunk_base}.tsv"

    ${launcher} python3 ${script_python} ${config_yaml} \
        --model ${xml_modelo} \
        --points ${tsv_pedaco} \
        --landscape "\${partial_file}"
    """
}


process CONSOLIDAR_LANDSCAPE {

    publishDir "${params.workflow.outdir}", mode: 'copy'

    input:
    path resultados_parciais
    val landscape_name

    output:
    path "${landscape_name}", emit: landscape_final

    script:
    """
    : > ${landscape_name}

    for f in \$(printf '%s\n' resultado_*.tsv | sort); do
        cat "\$f" >> ${landscape_name}
    done
    """
}


process PLOTAR_GRAFICO {

    publishDir "${params.workflow.outdir}", mode: 'copy'

    input:
    path landscape_tsv
    path script_python
    path config_yaml
    val np_mpi
    val matrix_name
    val figure_name
    val activate_cmd

    output:
    path "${figure_name}", emit: figura
    path "${matrix_name}", emit: matriz

    script:
    """
    ${activate_cmd}

    mpirun -np ${np_mpi} python3 ${script_python} ${config_yaml} \
        --landscape ${landscape_tsv} \
        --matrix ${matrix_name} \
        --figure ${figure_name}
    """
}


workflow {

    // =====================================================
    // VALIDAÇÃO DA CONFIGURAÇÃO
    // =====================================================

    if (params.workflow == null) {
        error "ERRO: seção 'workflow' não encontrada no arquivo YAML."
    }

    if (params.workflow.scripts == null) {
        error "ERRO: seção 'workflow.scripts' não encontrada no arquivo YAML."
    }

    def required_values = [
        "workflow.config_file"                 : params.workflow.config_file,
        "workflow.scripts.stationary_points"  : params.workflow.scripts.stationary_points,
        "workflow.scripts.simulation"         : params.workflow.scripts.simulation,
        "workflow.scripts.draw_landscape"     : params.workflow.scripts.draw_landscape,
        "workflow.outdir"                     : params.workflow.outdir,
        "workflow.chunk_size"                  : params.workflow.chunk_size,
        "workflow.np_mpi"                      : params.workflow.np_mpi,
        "model.file"                           : params.model?.file,
        "stationary_points.output"             : params.stationary_points?.output,
        "output.landscape"                     : params.output?.landscape,
        "output.matrix"                        : params.output?.matrix,
        "output.figure"                        : params.output?.figure
    ]

    def missing = required_values.findAll { key, value ->
        value == null || value.toString().trim().isEmpty()
    }.keySet()

    if (!missing.isEmpty()) {
        error """
        ERRO: faltam parâmetros obrigatórios no arquivo YAML:

          ${missing.join('\n          ')}
        """
    }


    // =====================================================
    // PARÂMETROS DE EXECUÇÃO
    // =====================================================

    int chunk_size = params.workflow.chunk_size as int
    int np_mpi     = params.workflow.np_mpi as int

    if (chunk_size < 1) {
        error "ERRO: workflow.chunk_size deve ser >= 1."
    }

    if (np_mpi < 1) {
        error "ERRO: workflow.np_mpi deve ser >= 1."
    }

    // activate_cmd e launcher são opcionais
    def activate_cmd = params.workflow.activate_cmd
        ? params.workflow.activate_cmd.toString()
        : "true"

    def launcher = params.workflow.launcher != null
        ? params.workflow.launcher.toString().trim()
        : ""


    // =====================================================
    // NOMES DAS SAÍDAS
    // =====================================================

    def points_output_name = new File(
        params.stationary_points.output.toString()
    ).getName()

    def landscape_name = new File(
        params.output.landscape.toString()
    ).getName()

    def matrix_name = new File(
        params.output.matrix.toString()
    ).getName()

    def figure_name = new File(
        params.output.figure.toString()
    ).getName()


    // =====================================================
    // CHANNELS DE ENTRADA
    // =====================================================

    config_ch = channel
        .fromPath(params.workflow.config_file, checkIfExists: true)
        .first()

    script_pontos_ch = channel
        .fromPath(
            params.workflow.scripts.stationary_points,
            checkIfExists: true
        )
        .first()

    script_simulacao_ch = channel
        .fromPath(
            params.workflow.scripts.simulation,
            checkIfExists: true
        )
        .first()

    script_draw_ch = channel
        .fromPath(
            params.workflow.scripts.draw_landscape,
            checkIfExists: true
        )
        .first()

    xml_ch = channel
        .fromPath(
            params.model.file,
            checkIfExists: true
        )
        .first()


    // =====================================================
    // 1. ENCONTRAR PONTOS ESTACIONÁRIOS
    // =====================================================

    pontos_out = ENCONTRAR_PONTOS_ESTACIONARIOS(
        script_pontos_ch,
        config_ch,
        xml_ch,
        points_output_name,
        activate_cmd
    )


    // =====================================================
    // 2. DIVIDIR OS PONTOS EM CHUNKS
    // =====================================================

    pedacos_out = DIVIDIR_PONTOS(
        pontos_out.pontos_tsv,
        chunk_size
    )

    pedacos_ch = pedacos_out.pedacos_ch.flatten()


    // =====================================================
    // 3. SIMULAR CADA CHUNK NO COPASI
    // =====================================================

    simulacoes_out = GERAR_LANDSCAPE(
        pedacos_ch,
        script_simulacao_ch,
        config_ch,
        xml_ch,
        activate_cmd,
        launcher
    )


    // =====================================================
    // 4. CONSOLIDAR OS RESULTADOS
    // =====================================================

    parciais_list_ch = simulacoes_out.resultado_parcial.collect()

    consolidado_out = CONSOLIDAR_LANDSCAPE(
        parciais_list_ch,
        landscape_name
    )


    // =====================================================
    // 5. GERAR A SUPERFÍCIE
    // =====================================================

    PLOTAR_GRAFICO(
        consolidado_out.landscape_final,
        script_draw_ch,
        config_ch,
        np_mpi,
        matrix_name,
        figure_name,
        activate_cmd
    )
}
