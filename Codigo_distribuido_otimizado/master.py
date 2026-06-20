"""
Simulação Distribuída de Propagação de Fake News — MASTER
----------------------------------------------------------

Responsabilidades do master:
    1. Criar a grade inicial completa
    2. Dividir em fatias e enviar para cada worker (com ghost rows)
    3. A cada geração:
        a. Receber bordas de todos os workers
        b. Redistribuir como ghost rows
        c. Receber contagens parciais e somá-las
        d. Imprimir estatísticas
        e. Enviar sinal de continuar ou parar
    4. Ao final: solicitar fatias finais dos workers e gerar animação

Como rodar:
    python master.py --workers 2 --linhas 100 --colunas 100 --geracoes 80
    python master.py --workers 2 --linhas 100 --colunas 100 --geracoes 80 --animar
    python master.py --workers 2 --linhas 100 --colunas 100 --geracoes 80 --salvar propagacao.gif

    # Demo entre máquinas (master aceita conexões de qualquer interface):
    python master.py --workers 4 --host 0.0.0.0 --porta 5050 --linhas 350

Dependências:
    pip install psutil matplotlib pillow
"""

import socket
import pickle
import struct
import time
import tracemalloc
import argparse
import platform
import sys
import random

try:
    import psutil
except ImportError:
    psutil = None

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────
# Constantes (devem ser idênticas ao worker.py)
# ─────────────────────────────────────────────
HOST_PADRAO  = '0.0.0.0'   # aceita conexões de qualquer interface
PORTA_PADRAO = 65432
TIMEOUT_SEG  = 300         # rede-de-segurança: aborta se algo travar por > 5 min

IGNORANTE      = 0
ESPALHADOR     = 1
INATIVO        = 2

TIPO_NORMAL       = 0
TIPO_OBSTACULO    = 1
TIPO_FACT_CHECKER = 2

MAPA_CORES = ListedColormap(
    ['#FFFFFF', '#E63946', '#457B9D', '#7209B7', '#4A4A4A', '#007504']
)
# 0=ignorante, 1=afetado A, 2=afetado B, 3=colisão, 4=obstáculo, 5=fact-checker


# ─────────────────────────────────────────────
# Comunicação via socket
# ─────────────────────────────────────────────
def _receber_exato(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Worker desconectou inesperadamente.")
        buf += chunk
    return buf


def enviar(sock, objeto):
    # protocolo explícito (mais rápido e estável entre versões de Python)
    dados = pickle.dumps(objeto, protocol=pickle.HIGHEST_PROTOCOL)
    # uma única chamada sendall evita possível fragmentação em 2 pacotes TCP
    sock.sendall(struct.pack('>I', len(dados)) + dados)


def receber(sock):
    cabecalho = _receber_exato(sock, 4)
    tamanho   = struct.unpack('>I', cabecalho)[0]
    return pickle.loads(_receber_exato(sock, tamanho))


# ─────────────────────────────────────────────
# Classe Indivíduo
# ─────────────────────────────────────────────
class Individuo:
    __slots__ = ['estado_A', 'mutacao_A', 'estado_B', 'mutacao_B', 'tipo']

    def __init__(self):
        self.estado_A  = IGNORANTE
        self.estado_B  = IGNORANTE
        self.mutacao_A = 0
        self.mutacao_B = 0
        self.tipo      = TIPO_NORMAL

    def clonar(self):
        novo           = Individuo()
        novo.estado_A  = self.estado_A
        novo.mutacao_A = self.mutacao_A
        novo.estado_B  = self.estado_B
        novo.mutacao_B = self.mutacao_B
        novo.tipo      = self.tipo
        return novo


# ─────────────────────────────────────────────
# Criação da grade
# ─────────────────────────────────────────────
def criar_grade(linhas, colunas,
                perc_espalhadores=0.02,
                perc_obstaculos=0.08,
                perc_cura=0.002,
                semente=15):
    rng_init = random.Random(semente)
    grade    = [[Individuo() for _ in range(colunas)] for _ in range(linhas)]
    total    = linhas * colunas

    for _ in range(int(total * perc_obstaculos)):
        grade[rng_init.randint(0, linhas-1)][rng_init.randint(0, colunas-1)].tipo = TIPO_OBSTACULO
    for _ in range(int(total * perc_cura)):
        grade[rng_init.randint(0, linhas-1)][rng_init.randint(0, colunas-1)].tipo = TIPO_FACT_CHECKER

    n_por_news = int((total * perc_espalhadores) / 2)
    c = 0
    while c < n_por_news:
        i = rng_init.randint(0, int(linhas * 0.3))
        j = rng_init.randint(0, int(colunas * 0.3))
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_A == IGNORANTE:
            grade[i][j].estado_A  = ESPALHADOR
            grade[i][j].mutacao_A = 1
            c += 1

    c = 0
    while c < n_por_news:
        i = rng_init.randint(int(linhas * 0.7), linhas-1)
        j = rng_init.randint(int(colunas * 0.7), colunas-1)
        if grade[i][j].tipo == TIPO_NORMAL and grade[i][j].estado_B == IGNORANTE:
            grade[i][j].estado_B  = ESPALHADOR
            grade[i][j].mutacao_B = 1
            c += 1

    return grade


# ─────────────────────────────────────────────
# Divisão da grade em fatias com ghost rows
# ─────────────────────────────────────────────
def dividir_grade(grade, num_workers):
    total_linhas = len(grade)
    base, extra  = divmod(total_linhas, num_workers)
    fatias = []
    pos    = 0
    for idx in range(num_workers):
        tam = base + (1 if idx < extra else 0)
        fim = pos + tam
        ghost_topo = grade[pos - 1] if pos > 0           else None
        ghost_base = grade[fim]     if fim < total_linhas else None
        fatias.append({
            'fatia'       : [ghost_topo] + grade[pos:fim] + [ghost_base],
            'linha_inicio': pos,
            'num_linhas'  : tam,
        })
        pos = fim
    return fatias


# ─────────────────────────────────────────────
# Redução das contagens parciais
# ─────────────────────────────────────────────
def somar_contagens(lista_contagens):
    total = {
        'ignorantes'    : 0,
        'afetados_A'    : 0,
        'afetados_B'    : 0,
        'colisoes'      : 0,
        'obstaculos'    : 0,
        'fact_checkers' : 0,
        'espalhadores_A': 0,
        'espalhadores_B': 0,
    }
    for c in lista_contagens:
        for chave in total:
            total[chave] += c.get(chave, 0)
    return total


# ─────────────────────────────────────────────
# Visualização
# ─────────────────────────────────────────────
def grade_para_matriz_visual(grade):
    """
    Converte grade de Individuo em matriz de inteiros 0-5 para visualização:
      0 = ignorante puro
      1 = afetado só por A  (vermelho)
      2 = afetado só por B  (azul)
      3 = colisão A+B       (roxo)
      4 = obstáculo         (cinza escuro)
      5 = fact-checker      (verde)
    """
    linhas  = len(grade)
    colunas = len(grade[0])
    matriz  = [[0] * colunas for _ in range(linhas)]
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
                elif a_inf:
                    matriz[i][j] = 1
                elif b_inf:
                    matriz[i][j] = 2
    return matriz


def reconstruir_grade_de_fatias(fatias_recebidas, linhas, colunas):
    """
    Reconstrói a grade completa a partir das fatias reais enviadas pelos workers.
    Cada fatia contém apenas as linhas reais (sem ghost rows).
    """
    grade = []
    for fatia in fatias_recebidas:
        grade.extend(fatia)
    return grade


def gerar_animacao(historico_visuais, historico_stats, linhas, colunas,
                   num_workers, salvar=None):
    """
    Gera animação da evolução da grade ao longo das gerações.
    historico_visuais : lista de matrizes visuais (uma por geração)
    historico_stats   : lista de dicts de contagem (uma por geração)
    salvar            : caminho .gif ou .mp4, ou None para exibir janela
    """
    if not historico_visuais:
        print("Histórico vazio — animação não gerada.")
        return

    n_ger = len(historico_visuais)
    print(f"\nGerando animação ({n_ger} frames)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Painel esquerdo: grade ─────────────────────────────────────
    ax_grade = axes[0]
    ax_grade.set_facecolor('#1a1a2e')
    ax_grade.set_title('Estado da População', color='white', fontsize=12, pad=10)
    ax_grade.axis('off')

    img = ax_grade.imshow(
        historico_visuais[0],
        cmap=MAPA_CORES, vmin=0, vmax=5,
        interpolation='nearest'
    )

    # Legenda manual
    from matplotlib.patches import Patch
    legenda = [
        Patch(color='#FFFFFF', label='Ignorante'),
        Patch(color='#E63946', label='Fake News A'),
        Patch(color='#457B9D', label='Fake News B'),
        Patch(color='#7209B7', label='Colisão A+B'),
        Patch(color='#4A4A4A', label='Obstáculo'),
        Patch(color='#007504', label='Fact-Checker'),
    ]
    ax_grade.legend(
        handles=legenda, loc='lower center',
        bbox_to_anchor=(0.5, -0.12), ncol=3,
        fontsize=8, facecolor='#1a1a2e', labelcolor='white',
        framealpha=0.8
    )

    titulo_grade = ax_grade.set_title(
        f'Geração 1/{n_ger} — {num_workers} workers',
        color='white', fontsize=12, pad=10
    )

    # ── Painel direito: gráfico de linha das populações ───────────
    ax_stats = axes[1]
    ax_stats.set_facecolor('#1a1a2e')
    ax_stats.tick_params(colors='white')
    ax_stats.spines['bottom'].set_color('#555')
    ax_stats.spines['left'].set_color('#555')
    ax_stats.spines['top'].set_visible(False)
    ax_stats.spines['right'].set_visible(False)
    ax_stats.set_title('Evolução das Populações', color='white', fontsize=12, pad=10)
    ax_stats.set_xlabel('Geração', color='white', fontsize=9)
    ax_stats.set_ylabel('Indivíduos', color='white', fontsize=9)

    geracoes_eixo  = list(range(1, n_ger + 1))
    ignorantes_ser = [s['ignorantes']   for s in historico_stats]
    afetados_A_ser = [s['afetados_A']   for s in historico_stats]
    afetados_B_ser = [s['afetados_B']   for s in historico_stats]
    colisoes_ser   = [s['colisoes']     for s in historico_stats]
    fc_ser         = [s['fact_checkers'] for s in historico_stats]

    linha_ign, = ax_stats.plot([], [], color='#AAAAAA', linewidth=1.5, label='Ignorantes')
    linha_A,   = ax_stats.plot([], [], color='#E63946', linewidth=1.5, label='Fake News A')
    linha_B,   = ax_stats.plot([], [], color='#457B9D', linewidth=1.5, label='Fake News B')
    linha_col, = ax_stats.plot([], [], color='#7209B7', linewidth=1.5, label='Colisão')
    linha_fc,  = ax_stats.plot([], [], color='#007504', linewidth=1.5, label='Fact-Checker')

    ax_stats.set_xlim(1, n_ger)
    ax_stats.set_ylim(0, linhas * colunas)
    ax_stats.legend(
        fontsize=8, facecolor='#1a1a2e', labelcolor='white',
        framealpha=0.8, loc='upper right'
    )

    # Linha vertical marcando a geração atual no gráfico
    linha_atual = ax_stats.axvline(x=1, color='white', linewidth=0.8, alpha=0.5)

    plt.tight_layout(pad=2)

    def atualizar(frame):
        # Atualiza grade
        img.set_data(historico_visuais[frame])
        titulo_grade.set_text(
            f'Geração {frame+1}/{n_ger} — {num_workers} workers'
        )

        # Atualiza gráfico de linha até o frame atual
        ger_ate_agora = geracoes_eixo[:frame+1]
        linha_ign.set_data(ger_ate_agora, ignorantes_ser[:frame+1])
        linha_A.set_data(ger_ate_agora,   afetados_A_ser[:frame+1])
        linha_B.set_data(ger_ate_agora,   afetados_B_ser[:frame+1])
        linha_col.set_data(ger_ate_agora, colisoes_ser[:frame+1])
        linha_fc.set_data(ger_ate_agora,  fc_ser[:frame+1])
        linha_atual.set_xdata([frame+1, frame+1])

        return [img, titulo_grade, linha_ign, linha_A, linha_B,
                linha_col, linha_fc, linha_atual]

    anim = animation.FuncAnimation(
        fig, atualizar,
        frames=n_ger,
        interval=100,
        blit=True,
        repeat=False,
    )

    if salvar:
        print(f"Salvando em '{salvar}'...")
        if salvar.lower().endswith('.gif'):
            anim.save(salvar, writer='pillow', fps=10,
                      savefig_kwargs={'facecolor': '#1a1a2e'})
        else:
            anim.save(salvar, fps=10,
                      savefig_kwargs={'facecolor': '#1a1a2e'})
        plt.close(fig)
        print(f"Animação salva em '{salvar}'.")
    else:
        plt.show()


# ─────────────────────────────────────────────
# Exibição
# ─────────────────────────────────────────────
def imprimir_estatisticas(geracao, total, tempo_geracao=None):
    t = f"  [{tempo_geracao:.3f}s]" if tempo_geracao else ""
    print(
        f"Geração {geracao+1:03d}{t} | "
        f"Ign: {total['ignorantes']:>8,} | "
        f"A: {total['afetados_A']:>8,} | "
        f"B: {total['afetados_B']:>8,} | "
        f"Col: {total['colisoes']:>6,} | "
        f"FC: {total['fact_checkers']:>5,} | "
        f"Esp_A: {total['espalhadores_A']:>5,} | "
        f"Esp_B: {total['espalhadores_B']:>5,}"
    )


def gerar_relatorio_ambiente(num_workers, host, porta):
    print("=" * 65)
    print("  CONFIGURAÇÃO EXPERIMENTAL — AMBIENTE DE EXECUÇÃO (MASTER)")
    print("=" * 65)
    print(f"Sistema Operacional  : {platform.system()} {platform.release()}")
    print(f"Processador          : {platform.processor()}")
    if psutil:
        print(f"Núcleos (Físicos)    : {psutil.cpu_count(logical=False)}")
        print(f"Threads (Lógicos)    : {psutil.cpu_count(logical=True)}")
        print(f"Memória RAM Total    : {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Versão Python        : {sys.version.split()[0]}")
    print(f"Workers configurados : {num_workers}")
    print(f"Comunicação          : Sockets TCP — {host}:{porta}")
    print("=" * 65)


# ─────────────────────────────────────────────
# Conexão com workers
# ─────────────────────────────────────────────
def conectar_workers(num_workers, host, porta):
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    servidor.bind((host, porta))
    servidor.listen(num_workers)
    print(f"\nAguardando {num_workers} worker(s) em {host}:{porta}...")
    sockets = []
    for idx in range(num_workers):
        conn, addr = servidor.accept()
        # rede-de-segurança: aborta se algo travar por mais de TIMEOUT_SEG
        conn.settimeout(TIMEOUT_SEG)
        sockets.append(conn)
        print(f"  Worker {idx} conectado ({addr[0]}:{addr[1]})")
    servidor.close()
    print(f"Todos os {num_workers} workers conectados!\n")
    return sockets


# ─────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────
def executar_master(
    num_workers=2,
    linhas=100,
    colunas=100,
    geracoes=80,
    perc_espalhadores=0.02,
    perc_obstaculos=0.08,
    perc_cura=0.002,
    semente=15,
    animar=False,
    salvar_animacao=None,
    host=HOST_PADRAO,
    porta=PORTA_PADRAO,
):
    gerar_relatorio_ambiente(num_workers, host, porta)

    # ── 1. Cria grade completa ─────────────────────────────────────
    print("Criando grade inicial...")
    grade = criar_grade(
        linhas, colunas,
        perc_espalhadores=perc_espalhadores,
        perc_obstaculos=perc_obstaculos,
        perc_cura=perc_cura,
        semente=semente,
    )
    print(f"Grade {linhas}x{colunas} criada ({linhas*colunas:,} células).")

    # Guarda visual da geração 0 (estado inicial) para a animação
    visual_inicial = grade_para_matriz_visual(grade)

    # ── 2. Divide em fatias ────────────────────────────────────────
    fatias = dividir_grade(grade, num_workers)
    for idx, f in enumerate(fatias):
        print(f"  Worker {idx}: linhas {f['linha_inicio']} "
              f"até {f['linha_inicio'] + f['num_linhas'] - 1} "
              f"({f['num_linhas']} linhas reais)")

    # ── 3. Conecta aos workers ─────────────────────────────────────
    workers = conectar_workers(num_workers, host, porta)

    # ── 4. Envia configuração inicial ──────────────────────────────
    coletar_grade = animar or (salvar_animacao is not None)
    for idx, sock in enumerate(workers):
        enviar(sock, {
            'fatia'         : fatias[idx]['fatia'],
            'semente'       : semente,
            'linha_inicio'  : fatias[idx]['linha_inicio'],
            'num_linhas'    : fatias[idx]['num_linhas'],
            'colunas'       : colunas,
            'geracoes'      : geracoes,
            'idx_worker'    : idx,
            'coletar_grade' : coletar_grade,  # avisa worker se precisa enviar grade completa
        })
    print("Configuração inicial enviada para todos os workers.\n")

    # ── 5. Loop de gerações ────────────────────────────────────────
    print("=== SIMULAÇÃO DISTRIBUÍDA DE PROPAGAÇÃO DE FAKE NEWS ===\n")
    tracemalloc.start()
    tempo_total_inicio = time.perf_counter()

    # historico_stats começa com a contagem do estado INICIAL (geração 0),
    # casando 1:1 com historico_visuais. Sem isso, o gráfico de evolução
    # ficaria 1 frame mais curto que a animação da grade.
    plano_inicial = [c for linha in visual_inicial for c in linha]
    stats_inicial = {
        'ignorantes'    : plano_inicial.count(0),
        'afetados_A'    : plano_inicial.count(1),
        'afetados_B'    : plano_inicial.count(2),
        'colisoes'      : plano_inicial.count(3),
        'obstaculos'    : plano_inicial.count(4),
        'fact_checkers' : plano_inicial.count(5),
        # No estado inicial não há INATIVO ainda, então todos os "afetados"
        # ainda são espalhadores ativos.
        'espalhadores_A': plano_inicial.count(1),
        'espalhadores_B': plano_inicial.count(2),
    }
    historico_stats   = [stats_inicial]
    historico_visuais = [visual_inicial]  # começa com estado inicial

    geracao_final = 0
    for geracao in range(geracoes):
        t_ger = time.perf_counter()

        # Passo A: recebe bordas de TODOS (barreira de sincronização)
        bordas = [receber(sock) for sock in workers]

        # Passo B: redistribui ghost rows
        for idx, sock in enumerate(workers):
            ghost_topo = bordas[idx - 1]['base'] if idx > 0              else None
            ghost_base = bordas[idx + 1]['topo'] if idx < num_workers - 1 else None
            enviar(sock, {'topo': ghost_topo, 'base': ghost_base})

        # Passo C: recebe contagens parciais
        contagens = [receber(sock) for sock in workers]
        total     = somar_contagens(contagens)
        historico_stats.append(total)

        # Passo C2: se animação ativada, recebe também as fatias reais
        # para reconstruir a grade visual a cada geração
        if coletar_grade:
            fatias_recv = [receber(sock) for sock in workers]
            # Reconstrói grade completa das fatias (só linhas reais)
            grade_completa = []
            for fatia_linhas in fatias_recv:
                grade_completa.extend(fatia_linhas)
            historico_visuais.append(grade_para_matriz_visual(grade_completa))

        tempo_ger = time.perf_counter() - t_ger
        imprimir_estatisticas(geracao, total, tempo_ger)

        # Passo D: critério de parada
        sem_espalhadores = (
            total['espalhadores_A'] == 0 and
            total['espalhadores_B'] == 0
        )
        ultima_geracao = (geracao == geracoes - 1)
        deve_parar     = sem_espalhadores or ultima_geracao

        for sock in workers:
            enviar(sock, {'tipo': 'parar' if deve_parar else 'continuar'})

        geracao_final = geracao + 1
        if sem_espalhadores:
            print("\nPropagação encerrada: não há mais espalhadores ativos.")
            break
        if ultima_geracao:
            print("\nLimite de gerações atingido.")
            break

    tempo_total = time.perf_counter() - tempo_total_inicio
    _, pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── 6. Fecha conexões ──────────────────────────────────────────
    for sock in workers:
        sock.close()

    # ── 7. Resultado final ─────────────────────────────────────────
    print()
    print("=" * 50)
    print("  RESULTADO FINAL")
    print("=" * 50)
    print(f"Gerações executadas  : {geracao_final}")
    print(f"Tempo total          : {tempo_total:.4f} segundos")
    print(f"Pico de RAM (master) : {pico_bytes / (1024**2):.2f} MB")
    print(f"Workers utilizados   : {num_workers}")
    print("=" * 50)

    # ── 8. Animação ────────────────────────────────────────────────
    if coletar_grade:
        gerar_animacao(
            historico_visuais,
            historico_stats,
            linhas, colunas,
            num_workers,
            salvar=salvar_animacao,
        )

    return {
        'tempo_segundos': tempo_total,
        'pico_ram_mb'   : pico_bytes / (1024**2),
        'geracoes'      : geracao_final,
        'num_workers'   : num_workers,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Master — Simulação Distribuída Fake News')
    parser.add_argument('--workers',      type=int,   default=2,             help='Número de workers')
    parser.add_argument('--linhas',       type=int,   default=100,           help='Linhas da grade')
    parser.add_argument('--colunas',      type=int,   default=100,           help='Colunas da grade')
    parser.add_argument('--geracoes',     type=int,   default=80,            help='Máximo de gerações')
    parser.add_argument('--espalhadores', type=float, default=0.02,          help='%% inicial de espalhadores')
    parser.add_argument('--obstaculos',   type=float, default=0.08,          help='%% de obstáculos')
    parser.add_argument('--cura',         type=float, default=0.002,         help='%% de fact-checkers')
    parser.add_argument('--semente',      type=int,   default=15,            help='Semente RNG')
    parser.add_argument('--animar',       action='store_true',               help='Exibe animação ao final')
    parser.add_argument('--salvar',       type=str,   default=None,          help='Salva animação (.gif ou .mp4)')
    parser.add_argument('--host',         type=str,   default=HOST_PADRAO,
                        help=f'Interface de bind (default: {HOST_PADRAO} — aceita de qualquer máquina)')
    parser.add_argument('--porta',        type=int,   default=PORTA_PADRAO,
                        help=f'Porta TCP (default: {PORTA_PADRAO})')
    args = parser.parse_args()

    executar_master(
        num_workers       = args.workers,
        linhas            = args.linhas,
        colunas           = args.colunas,
        geracoes          = args.geracoes,
        perc_espalhadores = args.espalhadores,
        perc_obstaculos   = args.obstaculos,
        perc_cura         = args.cura,
        semente           = args.semente,
        animar            = args.animar,
        salvar_animacao   = args.salvar,
        host              = args.host,
        porta             = args.porta,
    )