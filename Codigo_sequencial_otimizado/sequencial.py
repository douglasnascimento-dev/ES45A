"""
Simulação Sequencial de Propagação de Fake News
-------------------------------------------------

Versão sequencial estendida, base para as versões Paralela (Threads) e
Distribuída (Sockets/RMI).

Estados base por indivíduo (por fake news A e B):
    0 = Ignorante
    1 = Espalhador
    2 = Inativo

Tipos de célula:
    NORMAL        = indivíduo comum, sujeito a propagação
    OBSTACULO     = célula morta (não processa, gera desbalanceamento real)
    FACT_CHECKER  = "agente da cura", neutraliza propagação na vizinhança

Decisões importantes desta versão:
    - RNG determinístico por célula/geração: garante que paralela e
      distribuída produzam EXATAMENTE o mesmo resultado do sequencial
      para a mesma semente.
    - Medição de pico real de RAM via tracemalloc.
    - Modo gráfico tolerante a ambiente (Jupyter ou script .py).
    - Histórico por geração disponível para análise/plot posterior.
"""

import random
import time
import platform
import sys
import os
import tracemalloc

try:
    import psutil
except ImportError:
    print("Aviso: 'psutil' não instalado. Métricas de hardware ficarão limitadas.")
    psutil = None

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

# ----------------------------------------------------------------------
# Constantes de estado e tipo
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
# Por que isso é necessário:
# Como introduzimos transições estocásticas (random.random() < p), se
# usarmos o RNG global do módulo `random`, a ordem de consumo do RNG
# muda quando paralelizamos (threads diferentes consomem em ordem
# imprevisível). Isso faria a versão paralela divergir do sequencial.
#
# Usando uma semente derivada determinística (semente_global, geração,
# i, j), cada célula tem seu próprio fluxo aleatório, INDEPENDENTE da
# ordem em que é processada. Resultado: paralela e distribuída produzem
# bit-a-bit o mesmo resultado do sequencial para a mesma semente.
#
# Importante: não usamos hash() do Python porque o PYTHONHASHSEED
# aleatoriza hashes de strings/tuplas entre execuções. Usamos mistura
# aritmética com primos grandes (estável entre processos e máquinas).
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
    """
    MELHORIA 1: Otimização Computacional (__slots__).
    Sem __dict__ por instância: economiza RAM e acelera acesso a
    atributos em grades extensas.

    MELHORIA 2: Múltiplas Fake News Simultâneas (A e B).
    Cada indivíduo carrega dois estados independentes, transformando
    a matriz em arena de colisão entre narrativas.

    MELHORIA 3: Mutação da Informação.
    Cada notícia carrega uma "força" (mutacao_X) que evolui ao longo
    das gerações e altera o poder de convencimento.
    """
    __slots__ = ['estado_A', 'mutacao_A', 'estado_B', 'mutacao_B', 'tipo']

    def __init__(self):
        self.estado_A = IGNORANTE
        self.estado_B = IGNORANTE
        self.mutacao_A = 0
        self.mutacao_B = 0
        self.tipo = TIPO_NORMAL

    def clonar(self):
        novo = Individuo()
        novo.estado_A = self.estado_A
        novo.mutacao_A = self.mutacao_A
        novo.estado_B = self.estado_B
        novo.mutacao_B = self.mutacao_B
        novo.tipo = self.tipo
        return novo


# ----------------------------------------------------------------------
# Ambiente / Relatório
# ----------------------------------------------------------------------
def em_jupyter():
    """Detecta se estamos rodando em um kernel Jupyter/IPython."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        # 'ZMQInteractiveShell' = Jupyter; 'TerminalInteractiveShell' = ipython no terminal
        return ip.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def gerar_relatorio_ambiente():
    """MELHORIA 4: coleta automática do hardware para o relatório experimental."""
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
    print("=" * 60)


# ----------------------------------------------------------------------
# Criação da grade inicial
# ----------------------------------------------------------------------
def criar_grade(
    linhas,
    colunas,
    perc_espalhadores=0.02,
    perc_obstaculos=0.08,
    perc_cura=0.002,
    semente=15,
):
    """
    Cria a matriz inicial.

    MELHORIA 5: Injeção de obstáculos (células mortas) -> força
    desbalanceamento real de carga, motivando balanceamento dinâmico
    na versão paralela/distribuída.
    """
    rng_init = random.Random(semente)  # RNG isolado, não polui o estado global
    grade = [[Individuo() for _ in range(colunas)] for _ in range(linhas)]
    total_celulas = linhas * colunas

    # Obstáculos
    for _ in range(int(total_celulas * perc_obstaculos)):
        grade[rng_init.randint(0, linhas - 1)][rng_init.randint(0, colunas - 1)].tipo = TIPO_OBSTACULO

    # Fact-checkers iniciais
    for _ in range(int(total_celulas * perc_cura)):
        grade[rng_init.randint(0, linhas - 1)][rng_init.randint(0, colunas - 1)].tipo = TIPO_FACT_CHECKER

    total_espalhadores_por_news = int((total_celulas * perc_espalhadores) / 2)

    # Espalhadores da Fake News A no canto superior esquerdo
    alocados_A = 0
    while alocados_A < total_espalhadores_por_news:
        i = rng_init.randint(0, int(linhas * 0.3))
        j = rng_init.randint(0, int(colunas * 0.3))
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_A == IGNORANTE:
            grade[i][j].estado_A = ESPALHADOR
            grade[i][j].mutacao_A = 1
            alocados_A += 1

    # Espalhadores da Fake News B no canto inferior direito
    alocados_B = 0
    while alocados_B < total_espalhadores_por_news:
        i = rng_init.randint(int(linhas * 0.7), linhas - 1)
        j = rng_init.randint(int(colunas * 0.7), colunas - 1)
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_B == IGNORANTE:
            grade[i][j].estado_B = ESPALHADOR
            grade[i][j].mutacao_B = 1
            alocados_B += 1

    return grade


# ----------------------------------------------------------------------
# Lógica de simulação
# ----------------------------------------------------------------------
def avaliar_vizinhanca(grade, i, j, tipo_news):
    """Conta espalhadores, agrega mutação média e detecta fact-checkers na vizinhança de Moore."""
    linhas, colunas = len(grade), len(grade[0])
    qtd_espalhadores, soma_mutacao, qtd_fact_checkers = 0, 0, 0

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < linhas and 0 <= nj < colunas:
                vizinho = grade[ni][nj]
                if vizinho.tipo == TIPO_FACT_CHECKER:
                    qtd_fact_checkers += 1
                    continue
                if vizinho.tipo == TIPO_OBSTACULO:
                    continue
                if tipo_news == 'A' and vizinho.estado_A == ESPALHADOR:
                    qtd_espalhadores += 1
                    soma_mutacao += vizinho.mutacao_A
                elif tipo_news == 'B' and vizinho.estado_B == ESPALHADOR:
                    qtd_espalhadores += 1
                    soma_mutacao += vizinho.mutacao_B

    mutacao_media = (soma_mutacao // qtd_espalhadores) if qtd_espalhadores > 0 else 0
    return qtd_espalhadores, mutacao_media, qtd_fact_checkers


def calcular_transicao(qtd_vizinhos, mutacao_media, qtd_fact_checkers, estado_concorrente, rng):
    """
    Decide se um ignorante vira espalhador.

    MELHORIA 3 (cont.): mutação alta reduz o limiar de convencimento.
    MELHORIA 6: Resistência cruzada - já foi infectado pela outra news -> mais difícil aderir.
    MELHORIA 7: probabilidade base de adesão de 65%.

    `rng` é um Random local determinístico (vindo de rng_celula).
    """
    if qtd_fact_checkers > 0:
        return IGNORANTE, 0

    indice_necessario = 1.75
    if mutacao_media >= 3:
        indice_necessario -= 1
    if mutacao_media >= 6:
        indice_necessario -= 2
    if estado_concorrente != IGNORANTE:
        indice_necessario += 3

    if qtd_vizinhos >= max(1, indice_necessario):
        if rng.random() < 0.65:
            return ESPALHADOR, mutacao_media + 1

    return IGNORANTE, 0


def proxima_geracao(grade, semente, geracao):
    """
    Calcula a próxima geração. `semente` e `geracao` são usadas para derivar
    o RNG local de cada célula (determinismo paralelo).
    """
    linhas, colunas = len(grade), len(grade[0])
    nova_grade = [[grade[i][j].clonar() for j in range(colunas)] for i in range(linhas)]

    for i in range(linhas):
        for j in range(colunas):
            ind_atual = grade[i][j]
            novo_ind = nova_grade[i][j]

            if ind_atual.tipo == TIPO_OBSTACULO or ind_atual.tipo == TIPO_FACT_CHECKER:
                continue

            rng = rng_celula(semente, geracao, i, j)

            viz_a, mut_a, cura_vizinhos = avaliar_vizinhanca(grade, i, j, 'A')

            # MELHORIA 8: contágio orgânico do fact-checking
            if cura_vizinhos >= 1 and rng.random() < 0.02:
                novo_ind.tipo = TIPO_FACT_CHECKER
                novo_ind.estado_A = IGNORANTE
                novo_ind.estado_B = IGNORANTE
                continue

            viz_b, mut_b, _ = avaliar_vizinhanca(grade, i, j, 'B')

            # Transições do estado A
            if ind_atual.estado_A == IGNORANTE:
                novo_ind.estado_A, novo_ind.mutacao_A = calcular_transicao(
                    viz_a, mut_a, cura_vizinhos, ind_atual.estado_B, rng
                )
            elif ind_atual.estado_A == ESPALHADOR:
                if rng.random() < (0.25 if cura_vizinhos > 0 else 0.5):
                    novo_ind.estado_A = INATIVO

            # Transições do estado B
            if ind_atual.estado_B == IGNORANTE:
                novo_ind.estado_B, novo_ind.mutacao_B = calcular_transicao(
                    viz_b, mut_b, cura_vizinhos, ind_atual.estado_A, rng
                )
            elif ind_atual.estado_B == ESPALHADOR:
                if rng.random() < (0.25 if cura_vizinhos > 0 else 0.5):
                    novo_ind.estado_B = INATIVO

    return nova_grade


# ----------------------------------------------------------------------
# Conversão para visualização e estatísticas
# ----------------------------------------------------------------------
def grade_para_matriz_visual(grade):
    """
    Converte a grade em uma matriz de inteiros 0-5:
      0 = ignorante puro
      1 = afetado só por A
      2 = afetado só por B
      3 = colisão (afetado por A e B)
      4 = obstáculo
      5 = fact-checker
    """
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
                a_inf = ind.estado_A in (ESPALHADOR, INATIVO)
                b_inf = ind.estado_B in (ESPALHADOR, INATIVO)
                if a_inf and b_inf:
                    matriz[i][j] = 3
                elif b_inf:
                    matriz[i][j] = 2
                elif a_inf:
                    matriz[i][j] = 1
    return matriz


def contar_estados_visual(visual):
    """Contagem dos 6 estados na matriz visual."""
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
    modo_grafico=False,
    salvar_animacao=None,   # caminho .gif ou .mp4 opcional, útil fora do Jupyter
    coletar_historico=True,
):
    """Executa a simulação sequencial completa."""
    grade = criar_grade(
        linhas, colunas,
        perc_espalhadores=perc_espalhadores,
        perc_obstaculos=perc_obstaculos,
        perc_cura=perc_cura,
        semente=semente,
    )

    if modo_grafico:
        return _executar_grafico(grade, geracoes, semente, linhas, colunas, salvar_animacao)
    else:
        return _executar_headless(grade, geracoes, semente, linhas, colunas, coletar_historico)


def _executar_headless(grade, geracoes, semente, linhas, colunas, coletar_historico):
    """
    MELHORIA 10: modo headless para benchmark - mede CPU e pico real de RAM.
    """
    print(f"\n[MODO HEADLESS] Benchmark CPU ({linhas}x{colunas}) - {geracoes} gerações")

    historico = []
    tracemalloc.start()
    tempo_inicio = time.perf_counter()

    for g in range(geracoes):
        grade = proxima_geracao(grade, semente, g)
        if coletar_historico:
            historico.append(contar_estados_visual(grade_para_matriz_visual(grade)))

    tempo_fim = time.perf_counter()
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    estatisticas = contar_estados_visual(grade_para_matriz_visual(grade))
    tempo_total = tempo_fim - tempo_inicio

    print("-" * 40)
    print(" RESULTADOS DO BENCHMARK ")
    print("-" * 40)
    print(f"Tempo Sequencial Puro : {tempo_total:.4f} segundos")
    print(f"Pico de RAM (alocada) : {pico_bytes / (1024 * 1024):.2f} MB")
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
        'pico_ram_mb': pico_bytes / (1024 * 1024),
        'estatisticas_finais': estatisticas,
        'historico': historico,
        'grade_final': grade,
    }


def _executar_grafico(grade, geracoes, semente, linhas, colunas, salvar_animacao):
    """
    MELHORIA 9: visualização gráfica.
    Funciona em Jupyter (retorna HTML embutido) e em script .py (chama plt.show()
    ou salva em arquivo se `salvar_animacao` for fornecido).
    """
    print(f"\n[MODO GRÁFICO] Renderizando matriz ({linhas}x{colunas}) - {geracoes} gerações...")
    tempo_inicio = time.perf_counter()

    estado_visual = grade_para_matriz_visual(grade)
    fig, ax = plt.subplots(figsize=(8, 8))
    matriz_plot = ax.imshow(estado_visual, cmap=MAPA_CORES, vmin=0, vmax=5)
    ax.axis('off')

    estado = {'geracao_atual': 0, 'grade_atual': grade}

    def atualizar_frame(_frame):
        if estado['geracao_atual'] < geracoes:
            estado['grade_atual'] = proxima_geracao(
                estado['grade_atual'], semente, estado['geracao_atual']
            )
            estado['geracao_atual'] += 1
            matriz_plot.set_data(grade_para_matriz_visual(estado['grade_atual']))
            ax.set_title(f"Geração {estado['geracao_atual']}/{geracoes}")
        return [matriz_plot]

    anim = animation.FuncAnimation(
        fig, atualizar_frame, frames=geracoes, interval=50, blit=False, repeat=False
    )

    if em_jupyter():
        # Jupyter: gera HTML embedável
        plt.close(fig)
        from IPython.display import HTML
        resultado = HTML(anim.to_jshtml())
        tempo_fim = time.perf_counter()
        print(f"Tempo de renderização (inclui UI): {tempo_fim - tempo_inicio:.4f}s")
        return resultado
    else:
        # Script: salva em arquivo ou exibe janela
        if salvar_animacao:
            print(f"Salvando animação em '{salvar_animacao}'...")
            if salvar_animacao.lower().endswith('.gif'):
                anim.save(salvar_animacao, writer='pillow', fps=20)
            else:
                anim.save(salvar_animacao, fps=20)
            plt.close(fig)
            tempo_fim = time.perf_counter()
            print(f"Animação salva. Tempo total: {tempo_fim - tempo_inicio:.4f}s")
        else:
            tempo_fim = time.perf_counter()
            print(f"Exibindo janela. Tempo de simulação: {tempo_fim - tempo_inicio:.4f}s")
            plt.show()
        return None


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gerar_relatorio_ambiente()

    # 1) Benchmark headless (dados para tabelas de speedup/eficiência)
    resultado = executar_simulacao(
        linhas=350,
        colunas=350,
        geracoes=150,
        modo_grafico=False,
    )

    # 2) Animação para apresentação - salva em GIF se rodar como script,
    #    ou retorna HTML inline se rodar em Jupyter (use display() lá).
    if em_jupyter():
        from IPython.display import display
        anim = executar_simulacao(
            linhas=150, colunas=150, geracoes=100, modo_grafico=True,
        )
        display(anim)
    else:
        executar_simulacao(
            linhas=150, colunas=150, geracoes=100,
            modo_grafico=True,
            salvar_animacao='propagacao.gif',
        )
