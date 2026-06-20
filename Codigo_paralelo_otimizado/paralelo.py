"""
Simulação com Multiprocessing de Propagação de Fake News
---------------------------------------------------------

Implementação adicional (proposta como MELHORIA/INOVAÇÃO):
comparação de paradigmas paralelos - threads vs processos em CPython.

Por que esta versão existe:
    A versão paralela com Threads é limitada pelo GIL do CPython - o
    bytecode Python é serializado mesmo com múltiplas threads. Esta
    versão usa multiprocessing.shared_memory para contornar o GIL
    completamente, com cada processo executando em paralelo real.

Diferenças técnicas em relação à versão com threads:
    - Cada worker é um processo independente, sem GIL compartilhado.
    - Comunicação via memória compartilhada (shared_memory) com dois
      buffers que alternam papel de leitura/escrita a cada geração.
      Custo de comunicação ~zero por geração (sem serialização).
    - Grade representada internamente como buffer plano de inteiros
      (5 atributos por célula), ao invés de objetos Individuo.

Garantias preservadas:
    - Determinismo: mesma semente -> mesmo resultado da versão
      sequencial, independente do número de processos.
      (RNG derivado por célula: rng_celula(semente, g, i, j))
    - Sem race condition: workers escrevem em faixas exclusivas de
      linhas do buffer de destino. Sem locks.
    - Paralelização explícita: divisão de carga manual por faixas
      de linhas, sem uso de bibliotecas de paralelização automática.
"""

import random
import time
import platform
import sys
import os
import tracemalloc
import multiprocessing as mp
from multiprocessing import shared_memory

try:
    import psutil
except ImportError:
    print("Aviso: 'psutil' não instalado. Métricas de hardware ficarão limitadas.")
    psutil = None

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
IGNORANTE = 0
ESPALHADOR = 1
INATIVO = 2

TIPO_NORMAL = 0
TIPO_OBSTACULO = 1
TIPO_FACT_CHECKER = 2

# Layout da célula no buffer plano: 5 inteiros consecutivos
OFF_EST_A = 0   # estado_A
OFF_MUT_A = 1   # mutacao_A
OFF_EST_B = 2   # estado_B
OFF_MUT_B = 3   # mutacao_B
OFF_TIPO  = 4   # tipo
N_ATTRS   = 5

MAPA_CORES = ListedColormap(
    ['#FFFFFF', '#E63946', '#457B9D', '#7209B7', '#4A4A4A', '#007504']
)

# ----------------------------------------------------------------------
# RNG determinístico por célula (mesmo do sequencial)
# ----------------------------------------------------------------------
_MASK64 = 0xFFFFFFFFFFFFFFFF

def rng_celula(semente, geracao, i, j):
    s = (
        (semente * 0x9E3779B97F4A7C15)
        ^ (geracao * 0xBF58476D1CE4E5B9)
        ^ (i * 0x94D049BB133111EB)
        ^ (j * 0xC2B2AE3D27D4EB4F)
    ) & _MASK64
    return random.Random(s)


# ----------------------------------------------------------------------
# Classe Individuo (mantida para API e comparabilidade com outras versões)
# ----------------------------------------------------------------------
class Individuo:
    __slots__ = ['estado_A', 'mutacao_A', 'estado_B', 'mutacao_B', 'tipo']

    def __init__(self):
        self.estado_A = IGNORANTE
        self.estado_B = IGNORANTE
        self.mutacao_A = 0
        self.mutacao_B = 0
        self.tipo = TIPO_NORMAL


# ----------------------------------------------------------------------
# Ambiente
# ----------------------------------------------------------------------
def em_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def gerar_relatorio_ambiente(num_processos=0):
    print("=" * 60)
    print(" CONFIGURAÇÃO EXPERIMENTAL - AMBIENTE DE EXECUÇÃO ")
    print("=" * 60)
    print(f"Sistema Operacional  : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processador          : {platform.processor()}")
    if psutil:
        print(f"Núcleos (Físicos)    : {psutil.cpu_count(logical=False)}")
        print(f"Threads (Lógicos)    : {psutil.cpu_count(logical=True)}")
        print(f"Memória RAM Total    : {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Versão Interpretador : {sys.version.split(' ')[0]}")
    print(f"Ambiente             : {'Jupyter/IPython' if em_jupyter() else 'Script (CPython)'}")
    if num_processos:
        print(f"Processos na Simulação: {num_processos}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Divisão de carga
# ----------------------------------------------------------------------
def _fatiar(total, n):
    base, extra = divmod(total, n)
    pos = 0
    for i in range(n):
        fim = pos + base + (1 if i < extra else 0)
        if pos < fim:
            yield pos, fim
        pos = fim


# ----------------------------------------------------------------------
# Criação da grade inicial (idêntica ao sequencial - garante reprodutibilidade)
# ----------------------------------------------------------------------
def criar_grade(linhas, colunas, perc_espalhadores=0.02,
                perc_obstaculos=0.08, perc_cura=0.002, semente=15):
    rng_init = random.Random(semente)
    grade = [[Individuo() for _ in range(colunas)] for _ in range(linhas)]
    total = linhas * colunas

    for _ in range(int(total * perc_obstaculos)):
        grade[rng_init.randint(0, linhas - 1)][rng_init.randint(0, colunas - 1)].tipo = TIPO_OBSTACULO
    for _ in range(int(total * perc_cura)):
        grade[rng_init.randint(0, linhas - 1)][rng_init.randint(0, colunas - 1)].tipo = TIPO_FACT_CHECKER

    n_por_news = int((total * perc_espalhadores) / 2)

    c = 0
    while c < n_por_news:
        i = rng_init.randint(0, int(linhas * 0.3))
        j = rng_init.randint(0, int(colunas * 0.3))
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_A == IGNORANTE:
            grade[i][j].estado_A = ESPALHADOR
            grade[i][j].mutacao_A = 1
            c += 1

    c = 0
    while c < n_por_news:
        i = rng_init.randint(int(linhas * 0.7), linhas - 1)
        j = rng_init.randint(int(colunas * 0.7), colunas - 1)
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_B == IGNORANTE:
            grade[i][j].estado_B = ESPALHADOR
            grade[i][j].mutacao_B = 1
            c += 1

    return grade


# ----------------------------------------------------------------------
# Conversões grade <-> buffer plano (apenas na borda da simulação)
# ----------------------------------------------------------------------
def _grade_para_buffer(grade, buf_view, linhas, colunas):
    for i in range(linhas):
        linha = grade[i]
        for j in range(colunas):
            base = (i * colunas + j) * N_ATTRS
            ind = linha[j]
            buf_view[base + OFF_EST_A] = ind.estado_A
            buf_view[base + OFF_MUT_A] = ind.mutacao_A
            buf_view[base + OFF_EST_B] = ind.estado_B
            buf_view[base + OFF_MUT_B] = ind.mutacao_B
            buf_view[base + OFF_TIPO]  = ind.tipo


def _buffer_para_grade(buf_view, linhas, colunas):
    grade = []
    for i in range(linhas):
        linha = []
        for j in range(colunas):
            base = (i * colunas + j) * N_ATTRS
            ind = Individuo()
            ind.estado_A  = buf_view[base + OFF_EST_A]
            ind.mutacao_A = buf_view[base + OFF_MUT_A]
            ind.estado_B  = buf_view[base + OFF_EST_B]
            ind.mutacao_B = buf_view[base + OFF_MUT_B]
            ind.tipo      = buf_view[base + OFF_TIPO]
            linha.append(ind)
        grade.append(linha)
    return grade


# ----------------------------------------------------------------------
# Globais do processo worker
# ----------------------------------------------------------------------
# Cada processo worker mantém uma referência aos dois shared_memory blocks
# e as views correspondentes. Setadas pelo initializer do Pool.
_w_shm_a = None
_w_shm_b = None
_w_view_a = None
_w_view_b = None
_w_linhas = 0
_w_colunas = 0


def _init_worker_processo(name_a, name_b, linhas, colunas):
    """Initializer do Pool: cada processo abre as shared memories pelo nome."""
    global _w_shm_a, _w_shm_b, _w_view_a, _w_view_b, _w_linhas, _w_colunas
    _w_shm_a = shared_memory.SharedMemory(name=name_a)
    _w_shm_b = shared_memory.SharedMemory(name=name_b)
    _w_view_a = memoryview(_w_shm_a.buf).cast('i')   # int32
    _w_view_b = memoryview(_w_shm_b.buf).cast('i')
    _w_linhas = linhas
    _w_colunas = colunas


# ----------------------------------------------------------------------
# Lógica de transição (operando diretamente sobre o buffer plano)
# ----------------------------------------------------------------------
def _avaliar_vizinhanca_buf(src, i, j, linhas, colunas, ler_estado_off, ler_mut_off):
    """
    Conta espalhadores e fact-checkers na vizinhança de Moore, lendo direto
    do buffer plano. `ler_estado_off`/`ler_mut_off` selecionam A (0,1) ou B (2,3).
    """
    n_esp = n_fact = soma_mut = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if not (0 <= ni < linhas and 0 <= nj < colunas):
                continue
            nbase = (ni * colunas + nj) * N_ATTRS
            n_tipo = src[nbase + OFF_TIPO]
            if n_tipo == TIPO_FACT_CHECKER:
                n_fact += 1
            elif n_tipo != TIPO_OBSTACULO:
                if src[nbase + ler_estado_off] == ESPALHADOR:
                    n_esp += 1
                    soma_mut += src[nbase + ler_mut_off]
    mut_media = (soma_mut // n_esp) if n_esp else 0
    return n_esp, mut_media, n_fact


def _calcular_transicao(n_viz, mut_media, n_fact, estado_conc, rng):
    if n_fact > 0:
        return IGNORANTE, 0
    limiar = 1.75
    if mut_media >= 3:
        limiar -= 1.0
    if mut_media >= 6:
        limiar -= 2.0
    if estado_conc != IGNORANTE:
        limiar += 3.0
    if n_viz >= max(1, limiar) and rng.random() < 0.65:
        return ESPALHADOR, mut_media + 1
    return IGNORANTE, 0


# ----------------------------------------------------------------------
# Worker principal: processa linhas [i0, i1)
# ----------------------------------------------------------------------
# Sem race condition:
#   - Leitura: apenas de `src` (buffer "atual" da geração)
#   - Escrita: apenas em `dst[linhas i0..i1)`, faixa exclusiva deste processo
#   - RNG: derivado de (semente, geração, i, j) - independente de quem processa
# ----------------------------------------------------------------------
def _worker_proc(args):
    i0, i1, semente, geracao, ler_de_a = args

    src = _w_view_a if ler_de_a else _w_view_b
    dst = _w_view_b if ler_de_a else _w_view_a
    linhas, colunas = _w_linhas, _w_colunas

    for i in range(i0, i1):
        for j in range(colunas):
            base = (i * colunas + j) * N_ATTRS

            # Leitura
            est_A = src[base + OFF_EST_A]
            mut_A = src[base + OFF_MUT_A]
            est_B = src[base + OFF_EST_B]
            mut_B = src[base + OFF_MUT_B]
            tipo  = src[base + OFF_TIPO]

            # Valores de saída (default = copiar)
            n_est_A, n_mut_A = est_A, mut_A
            n_est_B, n_mut_B = est_B, mut_B
            n_tipo = tipo

            if tipo != TIPO_OBSTACULO and tipo != TIPO_FACT_CHECKER:
                rng = rng_celula(semente, geracao, i, j)

                viz_a, mma, cura = _avaliar_vizinhanca_buf(
                    src, i, j, linhas, colunas, OFF_EST_A, OFF_MUT_A
                )

                converteu_em_fact = False
                if cura >= 1 and rng.random() < 0.02:
                    n_tipo = TIPO_FACT_CHECKER
                    n_est_A = IGNORANTE
                    n_mut_A = 0
                    n_est_B = IGNORANTE
                    n_mut_B = 0
                    converteu_em_fact = True

                if not converteu_em_fact:
                    viz_b, mmb, _ = _avaliar_vizinhanca_buf(
                        src, i, j, linhas, colunas, OFF_EST_B, OFF_MUT_B
                    )

                    # Transições estado A
                    if est_A == IGNORANTE:
                        n_est_A, n_mut_A = _calcular_transicao(viz_a, mma, cura, est_B, rng)
                    elif est_A == ESPALHADOR:
                        if rng.random() < (0.25 if cura > 0 else 0.5):
                            n_est_A = INATIVO

                    # Transições estado B
                    if est_B == IGNORANTE:
                        n_est_B, n_mut_B = _calcular_transicao(viz_b, mmb, cura, est_A, rng)
                    elif est_B == ESPALHADOR:
                        if rng.random() < (0.25 if cura > 0 else 0.5):
                            n_est_B = INATIVO

            # Escrita no buffer destino
            dst[base + OFF_EST_A] = n_est_A
            dst[base + OFF_MUT_A] = n_mut_A
            dst[base + OFF_EST_B] = n_est_B
            dst[base + OFF_MUT_B] = n_mut_B
            dst[base + OFF_TIPO]  = n_tipo


# ----------------------------------------------------------------------
# Visualização (idêntica às outras versões)
# ----------------------------------------------------------------------
def grade_para_matriz_visual(grade):
    linhas, colunas = len(grade), len(grade[0])
    matriz = [[0] * colunas for _ in range(linhas)]
    for i in range(linhas):
        for j in range(colunas):
            ind = grade[i][j]
            if ind.tipo == TIPO_OBSTACULO:
                matriz[i][j] = 4
            elif ind.tipo == TIPO_FACT_CHECKER:
                matriz[i][j] = 5
            else:
                a = ind.estado_A in (ESPALHADOR, INATIVO)
                b = ind.estado_B in (ESPALHADOR, INATIVO)
                if a and b:
                    matriz[i][j] = 3
                elif b:
                    matriz[i][j] = 2
                elif a:
                    matriz[i][j] = 1
    return matriz


def contar_estados_visual(visual):
    plano = [c for linha in visual for c in linha]
    return {
        'ignorantes': plano.count(0),
        'afetados_A': plano.count(1),
        'afetados_B': plano.count(2),
        'colisoes':   plano.count(3),
        'obstaculos': plano.count(4),
        'fact_checkers': plano.count(5),
    }


# ----------------------------------------------------------------------
# Execução principal
# ----------------------------------------------------------------------
def executar_simulacao(
    linhas=100,
    colunas=100,
    geracoes=80,
    perc_espalhadores=0.02,
    perc_obstaculos=0.08,
    perc_cura=0.002,
    semente=15,
    num_processos=None,
    modo_grafico=False,
    salvar_animacao=None,
):
    if num_processos is None:
        num_processos = os.cpu_count() or 4

    grade = criar_grade(
        linhas, colunas,
        perc_espalhadores=perc_espalhadores,
        perc_obstaculos=perc_obstaculos,
        perc_cura=perc_cura,
        semente=semente,
    )

    # Aloca dois blocos de shared memory: buffers A e B
    # A cada geração, workers leem de um e escrevem no outro (alternância).
    tamanho_bytes = linhas * colunas * N_ATTRS * 4   # int32 = 4 bytes
    shm_a = shared_memory.SharedMemory(create=True, size=tamanho_bytes)
    shm_b = shared_memory.SharedMemory(create=True, size=tamanho_bytes)

    try:
        view_a = memoryview(shm_a.buf).cast('i')
        view_b = memoryview(shm_b.buf).cast('i')

        # Estado inicial: A
        _grade_para_buffer(grade, view_a, linhas, colunas)
        # Zera B para evitar lixo (não estritamente necessário, mas limpo)
        for k in range(linhas * colunas * N_ATTRS):
            view_b[k] = 0

        # Pool com initializer que abre as shared memories em cada worker
        with mp.Pool(
            processes=num_processos,
            initializer=_init_worker_processo,
            initargs=(shm_a.name, shm_b.name, linhas, colunas),
        ) as pool:
            fatias = list(_fatiar(linhas, num_processos))

            if modo_grafico:
                return _executar_grafico(
                    pool, fatias, view_a, view_b,
                    linhas, colunas, geracoes, semente,
                    num_processos, salvar_animacao,
                )
            return _executar_headless(
                pool, fatias, view_a, view_b,
                linhas, colunas, geracoes, semente,
                num_processos,
            )

    finally:
        # Liberação obrigatória (especialmente importante no Windows)
        # Soltar views antes para evitar 'cannot close exported pointers exist'
        try: del view_a
        except Exception: pass
        try: del view_b
        except Exception: pass
        shm_a.close()
        shm_b.close()
        shm_a.unlink()
        shm_b.unlink()


def _executar_headless(pool, fatias, view_a, view_b,
                       linhas, colunas, geracoes, semente, num_processos):
    print(f"\n[MODO HEADLESS - MULTIPROCESSING] Benchmark CPU ({linhas}x{colunas}) - {geracoes} gerações | {num_processos} processos")

    tracemalloc.start()
    tempo_inicio = time.perf_counter()

    ler_de_a = True
    for g in range(geracoes):
        tarefas = [(i0, i1, semente, g, ler_de_a) for i0, i1 in fatias]
        # map (não imap) bloqueia até todas terminarem -> barreira natural
        pool.map(_worker_proc, tarefas)
        ler_de_a = not ler_de_a

    tempo_fim = time.perf_counter()
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Reconstrói a grade final a partir do buffer "atual"
    view_final = view_a if ler_de_a else view_b
    grade_final = _buffer_para_grade(view_final, linhas, colunas)
    visual = grade_para_matriz_visual(grade_final)
    estatisticas = contar_estados_visual(visual)
    tempo_total = tempo_fim - tempo_inicio

    # Saída no mesmo formato das outras versões
    print("-" * 40)
    print(" RESULTADOS DO BENCHMARK ")
    print("-" * 40)
    print(f"Tempo Multiproc. Puro: {tempo_total:.4f} segundos")
    print(f"Processos Utilizados : {num_processos}")
    print(f"Pico de RAM (alocada): {pico_bytes / (1024 * 1024):.2f} MB")
    print("Distribuição Final:")
    print(f"  - Ignorantes        : {estatisticas['ignorantes']}")
    print(f"  - Afetados por A    : {estatisticas['afetados_A']}")
    print(f"  - Afetados por B    : {estatisticas['afetados_B']}")
    print(f"  - Colisões (A+B)    : {estatisticas['colisoes']}")
    print(f"  - Obstáculos        : {estatisticas['obstaculos']}")
    print(f"  - Agentes da Cura   : {estatisticas['fact_checkers']}")
    print("-" * 40)

    return {
        'tempo_segundos': tempo_total,
        'num_processos': num_processos,
        'pico_ram_mb': pico_bytes / (1024 * 1024),
        'estatisticas_finais': estatisticas,
        'grade_final': grade_final,
    }


def _executar_grafico(pool, fatias, view_a, view_b,
                      linhas, colunas, geracoes, semente,
                      num_processos, salvar_animacao):
    print(f"\n[MODO GRÁFICO - MULTIPROCESSING] Renderizando ({linhas}x{colunas}) - {geracoes} gerações | {num_processos} processos...")
    tempo_inicio = time.perf_counter()

    estado = {'g': 0, 'ler_de_a': True}

    grade_atual = _buffer_para_grade(view_a, linhas, colunas)
    vis = grade_para_matriz_visual(grade_atual)
    fig, ax = plt.subplots(figsize=(8, 8))
    matriz_plot = ax.imshow(vis, cmap=MAPA_CORES, vmin=0, vmax=5)
    ax.axis('off')

    def atualizar_frame(_frame):
        if estado['g'] < geracoes:
            tarefas = [(i0, i1, semente, estado['g'], estado['ler_de_a'])
                       for i0, i1 in fatias]
            pool.map(_worker_proc, tarefas)
            estado['ler_de_a'] = not estado['ler_de_a']
            estado['g'] += 1
            view_atual = view_a if estado['ler_de_a'] else view_b
            grade_atual = _buffer_para_grade(view_atual, linhas, colunas)
            matriz_plot.set_data(grade_para_matriz_visual(grade_atual))
            ax.set_title(f"Geração {estado['g']}/{geracoes}")
        return [matriz_plot]

    anim = animation.FuncAnimation(
        fig, atualizar_frame, frames=geracoes, interval=50, blit=False, repeat=False
    )

    if em_jupyter():
        plt.close(fig)
        from IPython.display import HTML
        resultado = HTML(anim.to_jshtml())
        print(f"Tempo de renderização (inclui UI): {time.perf_counter() - tempo_inicio:.4f}s")
        return resultado
    else:
        if salvar_animacao:
            print(f"Salvando animação em '{salvar_animacao}'...")
            if salvar_animacao.lower().endswith('.gif'):
                anim.save(salvar_animacao, writer='pillow', fps=20)
            else:
                anim.save(salvar_animacao, fps=20)
            plt.close(fig)
            print(f"Animação salva. Tempo total: {time.perf_counter() - tempo_inicio:.4f}s")
        else:
            print(f"Exibindo janela. Tempo de simulação: {time.perf_counter() - tempo_inicio:.4f}s")
            plt.show()
        return None


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANTE no Windows: o guard __main__ é obrigatório para
    # multiprocessing com spawn (que é o default no Windows).
    mp.freeze_support()

    N = os.cpu_count() or 4
    gerar_relatorio_ambiente(num_processos=N)

    # Mesma carga das outras versões - resultados diretamente comparáveis no PDF
    resultado = executar_simulacao(
        linhas=50,
        colunas=50,
        geracoes=150,
        num_processos=N,
        modo_grafico=False,
    )

    # Animação para apresentação
    if em_jupyter():
        from IPython.display import display
        anim = executar_simulacao(
            linhas=150, colunas=150, geracoes=100,
            num_processos=N, modo_grafico=True,
        )
        display(anim)
    else:
        executar_simulacao(
            linhas=150, colunas=150, geracoes=100,
            num_processos=N, modo_grafico=True,
            salvar_animacao='propagacao_multiprocessing.gif',
        )