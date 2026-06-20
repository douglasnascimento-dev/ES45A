"""
Simulação Paralela de Propagação de Fake News
----------------------------------------------

Versão paralela com Threads (threading + ThreadPoolExecutor).
Mantém a mesma lógica de simulação da versão sequencial, dividindo o
trabalho de cada geração entre N threads por faixas de linhas.

Garantias importantes:
    - Reprodutibilidade exata: para a mesma semente, qualquer número de
      threads produz o mesmo resultado final da versão sequencial.
      (Determinismo via RNG derivado por célula: rng_celula(semente, g, i, j))
    - Ausência de race condition: cada thread lê do snapshot imutável
      (`grade`) e escreve apenas em sua faixa exclusiva de `nova_grade`.
      Nenhum lock é necessário.
    - Paralelização explícita: divisão de carga implementada manualmente
      por faixas de linhas, sem uso de bibliotecas de paralelização
      automática (NumPy vetorizado, Numba, etc.).
"""

import random
import time
import platform
import sys
import os
import tracemalloc
import threading
from concurrent.futures import ThreadPoolExecutor

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

MAPA_CORES = ListedColormap(
    ['#FFFFFF', '#E63946', '#457B9D', '#7209B7', '#4A4A4A', '#007504']
)


# ----------------------------------------------------------------------
# RNG determinístico por célula
# ----------------------------------------------------------------------
# Mesma função do sequencial. Garante que, para a mesma semente, qualquer
# ordem de processamento (1 thread, 8 threads, embaralhado) produza o
# mesmo resultado. Sem isso, threads paralelas consumiriam o RNG global
# em ordem imprevisível e a versão paralela divergiria do sequencial.
# ----------------------------------------------------------------------
_MASK64 = 0xFFFFFFFFFFFFFFFF

def rng_celula(semente, geracao, i, j):
    """Retorna um random.Random determinístico para a célula (i,j) na geração dada."""
    s = (
        (semente * 0x9E3779B97F4A7C15)
        ^ (geracao * 0xBF58476D1CE4E5B9)
        ^ (i * 0x94D049BB133111EB)
        ^ (j * 0xC2B2AE3D27D4EB4F)
    ) & _MASK64
    return random.Random(s)


# ----------------------------------------------------------------------
# Classe Indivíduo
# ----------------------------------------------------------------------
class Individuo:
    __slots__ = ['estado_A', 'mutacao_A', 'estado_B', 'mutacao_B', 'tipo']

    def __init__(self):
        self.estado_A = IGNORANTE
        self.estado_B = IGNORANTE
        self.mutacao_A = 0
        self.mutacao_B = 0
        self.tipo = TIPO_NORMAL

    def clonar(self):
        n = Individuo()
        n.estado_A = self.estado_A
        n.mutacao_A = self.mutacao_A
        n.estado_B = self.estado_B
        n.mutacao_B = self.mutacao_B
        n.tipo = self.tipo
        return n


# ----------------------------------------------------------------------
# Ambiente / Relatório
# ----------------------------------------------------------------------
def em_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def gerar_relatorio_ambiente(num_threads=0):
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
    if num_threads:
        print(f"Threads na Simulação : {num_threads}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Divisão de carga entre threads
# ----------------------------------------------------------------------
def _fatiar(total, n):
    """
    Divide [0, total) em n fatias contíguas, distribuindo o resto da divisão
    como +1 linha nas primeiras fatias. Balanceamento máximo, nenhuma fatia ociosa.
    """
    base, extra = divmod(total, n)
    pos = 0
    for i in range(n):
        fim = pos + base + (1 if i < extra else 0)
        if pos < fim:
            yield pos, fim
        pos = fim


# ----------------------------------------------------------------------
# Criação da grade inicial
# ----------------------------------------------------------------------
# Mantida estritamente serial e idêntica ao sequencial. Paralelizar a
# inicialização aqui não compensa (criação de Individuos é barata e ocorre
# apenas uma vez), e qualquer divergência aqui quebraria a reprodutibilidade
# com o sequencial.
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

    alocados = 0
    while alocados < n_por_news:
        i = rng_init.randint(0, int(linhas * 0.3))
        j = rng_init.randint(0, int(colunas * 0.3))
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_A == IGNORANTE:
            grade[i][j].estado_A = ESPALHADOR
            grade[i][j].mutacao_A = 1
            alocados += 1

    alocados = 0
    while alocados < n_por_news:
        i = rng_init.randint(int(linhas * 0.7), linhas - 1)
        j = rng_init.randint(int(colunas * 0.7), colunas - 1)
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_B == IGNORANTE:
            grade[i][j].estado_B = ESPALHADOR
            grade[i][j].mutacao_B = 1
            alocados += 1

    return grade


# ----------------------------------------------------------------------
# Lógica pura da simulação (funções compartilhadas pelos workers)
# ----------------------------------------------------------------------
def _avaliar_vizinhanca(grade, i, j, tipo_news):
    linhas, colunas = len(grade), len(grade[0])
    n_esp = n_fact = soma_mut = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if not (0 <= ni < linhas and 0 <= nj < colunas):
                continue
            viz = grade[ni][nj]
            if viz.tipo == TIPO_FACT_CHECKER:
                n_fact += 1
            elif viz.tipo != TIPO_OBSTACULO:
                if tipo_news == 'A' and viz.estado_A == ESPALHADOR:
                    n_esp += 1
                    soma_mut += viz.mutacao_A
                elif tipo_news == 'B' and viz.estado_B == ESPALHADOR:
                    n_esp += 1
                    soma_mut += viz.mutacao_B
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
# Workers de thread
# ----------------------------------------------------------------------
# Cada worker recebe uma faixa exclusiva de linhas [i0, i1).
#
# Sem race condition:
#   - Leitura: apenas de `grade` (snapshot imutável durante a geração)
#   - Escrita: apenas em `nova_grade[i0..i1-1]` (fatia disjunta de cada thread)
#   - RNG:    derivado de (semente, geração, i, j) - independente de qual
#             thread executa
# ----------------------------------------------------------------------
def _worker_clonar(grade, nova_grade, i0, i1):
    colunas = len(grade[0])
    for i in range(i0, i1):
        linha_origem = grade[i]
        nova_grade[i] = [linha_origem[j].clonar() for j in range(colunas)]


def _worker_processar(grade, nova_grade, i0, i1, semente, geracao):
    colunas = len(grade[0])
    for i in range(i0, i1):
        for j in range(colunas):
            ind = grade[i][j]
            novo = nova_grade[i][j]

            if ind.tipo == TIPO_OBSTACULO or ind.tipo == TIPO_FACT_CHECKER:
                continue

            rng = rng_celula(semente, geracao, i, j)

            viz_a, mut_a, cura = _avaliar_vizinhanca(grade, i, j, 'A')

            # Propagação orgânica de fact-checkers
            if cura >= 1 and rng.random() < 0.02:
                novo.tipo = TIPO_FACT_CHECKER
                novo.estado_A = IGNORANTE
                novo.estado_B = IGNORANTE
                continue

            viz_b, mut_b, _ = _avaliar_vizinhanca(grade, i, j, 'B')

            # Transições estado A
            if ind.estado_A == IGNORANTE:
                novo.estado_A, novo.mutacao_A = _calcular_transicao(
                    viz_a, mut_a, cura, ind.estado_B, rng
                )
            elif ind.estado_A == ESPALHADOR:
                if rng.random() < (0.25 if cura > 0 else 0.5):
                    novo.estado_A = INATIVO

            # Transições estado B
            if ind.estado_B == IGNORANTE:
                novo.estado_B, novo.mutacao_B = _calcular_transicao(
                    viz_b, mut_b, cura, ind.estado_A, rng
                )
            elif ind.estado_B == ESPALHADOR:
                if rng.random() < (0.25 if cura > 0 else 0.5):
                    novo.estado_B = INATIVO


def _worker_visualizar(grade, matriz, i0, i1):
    colunas = len(grade[0])
    for i in range(i0, i1):
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


# ----------------------------------------------------------------------
# Próxima geração (paralela em duas fases)
# ----------------------------------------------------------------------
def proxima_geracao(grade, executor, num_threads, semente, geracao):
    """
    Calcula a próxima geração em duas fases paralelas:
      Fase 1 - clonagem: cada thread copia sua fatia de grade -> nova_grade
      Fase 2 - transições: cada thread aplica as regras em sua fatia

    A barreira entre as fases (via .result()) garante snapshot consistente
    antes das transições começarem.
    """
    linhas = len(grade)
    nova_grade = [None] * linhas
    fatias = list(_fatiar(linhas, num_threads))

    # Fase 1: clonagem paralela
    futuros = [executor.submit(_worker_clonar, grade, nova_grade, a, b)
               for a, b in fatias]
    for f in futuros:
        f.result()

    # Fase 2: transições paralelas
    futuros = [executor.submit(_worker_processar, grade, nova_grade, a, b,
                               semente, geracao)
               for a, b in fatias]
    for f in futuros:
        f.result()

    return nova_grade


def grade_para_matriz_visual(grade, executor, num_threads):
    """Conversão para matriz visual em paralelo."""
    linhas, colunas = len(grade), len(grade[0])
    matriz = [[0] * colunas for _ in range(linhas)]
    futuros = [executor.submit(_worker_visualizar, grade, matriz, a, b)
               for a, b in _fatiar(linhas, num_threads)]
    for f in futuros:
        f.result()
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
    num_threads=None,
    modo_grafico=False,
    salvar_animacao=None,
    coletar_historico=True,
):
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    grade = criar_grade(
        linhas, colunas,
        perc_espalhadores=perc_espalhadores,
        perc_obstaculos=perc_obstaculos,
        perc_cura=perc_cura,
        semente=semente,
    )

    # Pool criado uma única vez: as threads ficam ativas durante todas as
    # gerações, eliminando o custo de spawn/join por geração.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        if modo_grafico:
            return _executar_grafico(
                grade, geracoes, semente, linhas, colunas,
                executor, num_threads, salvar_animacao,
            )
        return _executar_headless(
            grade, geracoes, semente, linhas, colunas,
            executor, num_threads, coletar_historico,
        )


def _executar_headless(grade, geracoes, semente, linhas, colunas,
                       executor, num_threads, coletar_historico):
    print(f"\n[MODO HEADLESS - PARALELO] Benchmark CPU ({linhas}x{colunas}) - {geracoes} gerações | {num_threads} threads")

    historico = []
    tracemalloc.start()
    tempo_inicio = time.perf_counter()

    for g in range(geracoes):
        grade = proxima_geracao(grade, executor, num_threads, semente, g)
        if coletar_historico:
            historico.append(contar_estados_visual(
                grade_para_matriz_visual(grade, executor, num_threads)
            ))

    tempo_fim = time.perf_counter()
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    visual = grade_para_matriz_visual(grade, executor, num_threads)
    estatisticas = contar_estados_visual(visual)
    tempo_total = tempo_fim - tempo_inicio

    # Saída no mesmo formato da versão sequencial, para facilitar comparação
    # direta no PDF.
    print("-" * 40)
    print(" RESULTADOS DO BENCHMARK ")
    print("-" * 40)
    print(f"Tempo Paralelo Puro  : {tempo_total:.4f} segundos")
    print(f"Threads Utilizadas   : {num_threads}")
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
        'num_threads': num_threads,
        'pico_ram_mb': pico_bytes / (1024 * 1024),
        'estatisticas_finais': estatisticas,
        'historico': historico,
        'grade_final': grade,
    }


def _executar_grafico(grade, geracoes, semente, linhas, colunas,
                      executor, num_threads, salvar_animacao):
    print(f"\n[MODO GRÁFICO - PARALELO] Renderizando ({linhas}x{colunas}) - {geracoes} gerações | {num_threads} threads...")
    tempo_inicio = time.perf_counter()

    estado_visual = grade_para_matriz_visual(grade, executor, num_threads)
    fig, ax = plt.subplots(figsize=(8, 8))
    matriz_plot = ax.imshow(estado_visual, cmap=MAPA_CORES, vmin=0, vmax=5)
    ax.axis('off')

    estado = {'geracao_atual': 0, 'grade_atual': grade}

    def atualizar_frame(_frame):
        if estado['geracao_atual'] < geracoes:
            estado['grade_atual'] = proxima_geracao(
                estado['grade_atual'], executor, num_threads,
                semente, estado['geracao_atual'],
            )
            estado['geracao_atual'] += 1
            matriz_plot.set_data(grade_para_matriz_visual(
                estado['grade_atual'], executor, num_threads
            ))
            ax.set_title(f"Geração {estado['geracao_atual']}/{geracoes}")
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
    N = os.cpu_count() or 4
    gerar_relatorio_ambiente(num_threads=N)

    # Mesma carga do sequencial -> resultados diretamente comparáveis no PDF.
    resultado = executar_simulacao(
        linhas=350,
        colunas=350,
        geracoes=150,
        num_threads=N,
        modo_grafico=False,
    )

    # Animação para a apresentação
    if em_jupyter():
        from IPython.display import display
        anim = executar_simulacao(
            linhas=150, colunas=150, geracoes=100,
            num_threads=N, modo_grafico=True,
        )
        display(anim)
    else:
        executar_simulacao(
            linhas=150, colunas=150, geracoes=100,
            num_threads=N, modo_grafico=True,
            salvar_animacao='propagacao_paralelo.gif',
        )
