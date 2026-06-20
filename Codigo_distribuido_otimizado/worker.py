"""
Simulação Distribuída de Propagação de Fake News — WORKER
----------------------------------------------------------

Responsabilidades do worker:
    1. Conectar ao master e receber sua fatia da grade
    2. A cada geração:
        a. Enviar suas linhas de borda (topo e base reais) para o master
        b. Receber ghost rows atualizadas do master
        c. Calcular proxima_geracao_local nas linhas reais
           (contando os estados durante o próprio cálculo — sem passagem extra)
        d. Enviar contagem parcial de estados
        e. Aguardar instrução de continuar ou parar

Como rodar (após iniciar o master):
    python worker.py                                  # conecta em localhost:65432
    python worker.py --host 192.168.1.10              # master em outra máquina
    python worker.py --host 192.168.1.10 --porta 5050 # porta customizada

    (cada worker em um terminal separado — o master os identifica
     pela ordem de chegada)
"""

import socket
import pickle
import struct
import time
import argparse
import random

# ─────────────────────────────────────────────
# Constantes (devem ser idênticas ao master.py)
# ─────────────────────────────────────────────
HOST_PADRAO  = 'localhost'
PORTA_PADRAO = 65432
TIMEOUT_SEG  = 300   # rede-de-segurança: se algo travar, abortar após 5 min

IGNORANTE      = 0
ESPALHADOR     = 1
INATIVO        = 2

TIPO_NORMAL       = 0
TIPO_OBSTACULO    = 1
TIPO_FACT_CHECKER = 2

_MASK64 = 0xFFFFFFFFFFFFFFFF


# ─────────────────────────────────────────────
# Funções de comunicação (idênticas ao master)
# ─────────────────────────────────────────────
def _receber_exato(sock, n):
    """Lê exatamente n bytes — obrigatório pois TCP pode fragmentar."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Master desconectou inesperadamente.")
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
# Classe Indivíduo (idêntica ao sequencial)
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
# RNG determinístico por célula
# ─────────────────────────────────────────────
def rng_celula(semente, geracao, i_global, j):
    """
    Gera um RNG isolado por célula usando o índice GLOBAL da linha.

    Por que índice global?
    O sequencial usa o índice da linha na grade completa. Se o worker
    usasse o índice local, o resultado divergiria do sequencial.

    Exemplo: worker 1 processa linhas globais 50-99.
    Ao processar sua linha local 1 (= linha global 50):
        ERRADO:  rng_celula(semente, g,  1, j)
        CORRETO: rng_celula(semente, g, 50, j)
    """
    s = (
        (semente   * 0x9E3779B97F4A7C15)
        ^ (geracao * 0xBF58476D1CE4E5B9)
        ^ (i_global * 0x94D049BB133111EB)
        ^ (j        * 0xC2B2AE3D27D4EB4F)
    ) & _MASK64
    return random.Random(s)


# ─────────────────────────────────────────────
# Lógica de simulação
# ─────────────────────────────────────────────
def avaliar_vizinhanca(grade_local, i_local, j, tipo_news):
    """
    Conta espalhadores e fact-checkers na vizinhança de Moore.

    grade_local inclui ghost rows nos índices 0 e -1, então acessar
    i_local-1 e i_local+1 nas bordas reais funciona naturalmente —
    a função não precisa saber que essas linhas vieram de outro worker.

    Verifica ambos os limites (vertical e horizontal) para não acessar
    linhas None (ghost de borda externa) ou colunas inexistentes.
    """
    linhas  = len(grade_local)
    colunas = len(grade_local[1])   # índice 1 = primeira linha real (nunca None)
    qtd_espalhadores = soma_mutacao = qtd_fact_checkers = 0

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni = i_local + di
            nj = j + dj

            # Verifica limites verticais E horizontais
            if not (0 <= ni < linhas and 0 <= nj < colunas):
                continue
            # Ghost row de borda externa (worker do topo ou do fundo)
            if grade_local[ni] is None:
                continue

            vizinho = grade_local[ni][nj]
            if vizinho.tipo == TIPO_FACT_CHECKER:
                qtd_fact_checkers += 1
                continue
            if vizinho.tipo == TIPO_OBSTACULO:
                continue
            if tipo_news == 'A' and vizinho.estado_A == ESPALHADOR:
                qtd_espalhadores += 1
                soma_mutacao     += vizinho.mutacao_A
            elif tipo_news == 'B' and vizinho.estado_B == ESPALHADOR:
                qtd_espalhadores += 1
                soma_mutacao     += vizinho.mutacao_B

    mutacao_media = (soma_mutacao // qtd_espalhadores) if qtd_espalhadores > 0 else 0
    return qtd_espalhadores, mutacao_media, qtd_fact_checkers


def calcular_transicao(qtd_vizinhos, mutacao_media, qtd_fact_checkers, estado_concorrente, rng):
    """
    Decide se um ignorante vira espalhador. Mantém todas as melhorias:
      - Fact-checker bloqueia completamente a transição
      - Mutação alta reduz o limiar de convencimento
      - Resistência cruzada: já infectado pela outra news dificulta adesão
      - Probabilidade base de 65%
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


def proxima_geracao_local(grade_local, semente, geracao, linha_inicio):
    """
    Calcula a próxima geração apenas para as linhas REAIS do worker
    e já retorna a contagem de estados (sem segunda passagem pela grade).

    Incorporado do código de referência: contar durante o cálculo
    elimina uma iteração completa sobre a grade por geração.

    Diferenças em relação ao sequencial:
      1. Itera range(1, len-1) — pula ghost rows (índices 0 e -1)
      2. Usa (linha_inicio + i_local - 1) como índice global no rng_celula
      3. Ghost rows são lidas mas nunca recalculadas
      4. Contagem de estados feita inline, durante o próprio loop
    """
    num_linhas_local = len(grade_local)
    colunas          = len(grade_local[1])

    # Clona apenas as linhas reais; ghost rows são mantidas por referência
    nova_grade       = [None] * num_linhas_local
    nova_grade[0]    = grade_local[0]    # ghost topo — não recalcula
    nova_grade[-1]   = grade_local[-1]   # ghost base — não recalcula

    # Contagem inline (incorporado do código de referência)
    contagem = {
        'ignorantes'    : 0,
        'afetados_A'    : 0,
        'afetados_B'    : 0,
        'colisoes'      : 0,
        'obstaculos'    : 0,
        'fact_checkers' : 0,
        'espalhadores_A': 0,
        'espalhadores_B': 0,
    }

    for i_local in range(1, num_linhas_local - 1):
        # Índice global — necessário para o rng_celula ser idêntico ao sequencial
        i_global   = linha_inicio + (i_local - 1)
        nova_linha = [grade_local[i_local][j].clonar() for j in range(colunas)]

        for j in range(colunas):
            ind_atual = grade_local[i_local][j]
            novo_ind  = nova_linha[j]

            # Obstáculos e fact-checkers não evoluem
            if ind_atual.tipo == TIPO_OBSTACULO:
                contagem['obstaculos'] += 1
                continue
            if ind_atual.tipo == TIPO_FACT_CHECKER:
                contagem['fact_checkers'] += 1
                continue

            rng = rng_celula(semente, geracao, i_global, j)

            viz_a, mut_a, cura = avaliar_vizinhanca(grade_local, i_local, j, 'A')

            # Contágio orgânico do fact-checking
            if cura >= 1 and rng.random() < 0.02:
                novo_ind.tipo    = TIPO_FACT_CHECKER
                novo_ind.estado_A = IGNORANTE
                novo_ind.estado_B = IGNORANTE
                nova_linha[j]    = novo_ind
                contagem['fact_checkers'] += 1
                continue

            viz_b, mut_b, _ = avaliar_vizinhanca(grade_local, i_local, j, 'B')

            # Transições estado A
            if ind_atual.estado_A == IGNORANTE:
                novo_ind.estado_A, novo_ind.mutacao_A = calcular_transicao(
                    viz_a, mut_a, cura, ind_atual.estado_B, rng
                )
            elif ind_atual.estado_A == ESPALHADOR:
                if rng.random() < (0.25 if cura > 0 else 0.5):
                    novo_ind.estado_A = INATIVO

            # Transições estado B
            if ind_atual.estado_B == IGNORANTE:
                novo_ind.estado_B, novo_ind.mutacao_B = calcular_transicao(
                    viz_b, mut_b, cura, ind_atual.estado_A, rng
                )
            elif ind_atual.estado_B == ESPALHADOR:
                if rng.random() < (0.25 if cura > 0 else 0.5):
                    novo_ind.estado_B = INATIVO

            nova_linha[j] = novo_ind

            # ── Contagem inline dos novos estados ──────────────────
            a_inf = novo_ind.estado_A in (ESPALHADOR, INATIVO)
            b_inf = novo_ind.estado_B in (ESPALHADOR, INATIVO)
            if a_inf and b_inf:
                contagem['colisoes']   += 1
            elif a_inf:
                contagem['afetados_A'] += 1
            elif b_inf:
                contagem['afetados_B'] += 1
            else:
                contagem['ignorantes'] += 1

            # Espalhadores ativos (critério de parada do master)
            if novo_ind.estado_A == ESPALHADOR:
                contagem['espalhadores_A'] += 1
            if novo_ind.estado_B == ESPALHADOR:
                contagem['espalhadores_B'] += 1

        nova_grade[i_local] = nova_linha

    return nova_grade, contagem


# ─────────────────────────────────────────────
# Loop principal do worker
# ─────────────────────────────────────────────
def executar_worker(host=HOST_PADRAO, porta=PORTA_PADRAO):
    print(f"[WORKER] Conectando ao master em {host}:{porta}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # rede-de-segurança: aborta se algo travar por mais de TIMEOUT_SEG
    sock.settimeout(TIMEOUT_SEG)

    # Tenta conectar com retries (master pode ainda estar inicializando)
    for tentativa in range(15):
        try:
            sock.connect((host, porta))
            break
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[WORKER] Master não disponível ainda, aguardando... ({tentativa+1}/15)")
            time.sleep(1)
    else:
        print("[WORKER] Não foi possível conectar ao master. Encerrando.")
        return

    print("[WORKER] Conectado!")

    # ── Recebe configuração inicial ────────────────────────────────
    config         = receber(sock)
    grade_local    = config['fatia']
    semente        = config['semente']
    linha_inicio   = config['linha_inicio']
    num_linhas     = config['num_linhas']
    idx_worker     = config['idx_worker']
    coletar_grade  = config.get('coletar_grade', False)

    print(f"[WORKER {idx_worker}] Fatia recebida: "
          f"linhas globais {linha_inicio} a {linha_inicio + num_linhas - 1} "
          f"({num_linhas} linhas reais)")

    # ── Loop de gerações ───────────────────────────────────────────
    geracao      = 0
    tempo_inicio = time.perf_counter()

    while True:
        # Passo 1: envia bordas reais ao master
        #   grade_local[1]  = primeira linha real
        #   grade_local[-2] = última linha real
        enviar(sock, {
            'topo': grade_local[1],
            'base': grade_local[-2],
        })

        # Passo 2: recebe ghost rows atualizadas do master
        ghosts          = receber(sock)
        grade_local[0]  = ghosts['topo']   # None se este worker é o do topo
        grade_local[-1] = ghosts['base']   # None se este worker é o do fundo

        # Passo 3: calcula próxima geração + contagem inline
        grade_local, contagem = proxima_geracao_local(
            grade_local, semente, geracao, linha_inicio
        )
        geracao += 1

        # Passo 4: envia contagem parcial ao master
        enviar(sock, contagem)

        # Passo 4b: se master pediu, envia também as linhas reais para animação
        # (apenas linhas reais: índices 1 até -2, sem ghost rows)
        if coletar_grade:
            linhas_reais = grade_local[1:-1]
            enviar(sock, linhas_reais)

        # Passo 5: aguarda instrução do master
        instrucao = receber(sock)
        if instrucao['tipo'] == 'parar':
            print(f"[WORKER {idx_worker}] Parada recebida na geração {geracao}.")
            break

    tempo_total = time.perf_counter() - tempo_inicio
    print(f"[WORKER {idx_worker}] Encerrado. "
          f"{geracao} gerações em {tempo_total:.4f}s.")
    sock.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker — Simulação Distribuída de Fake News'
    )
    parser.add_argument('--host',  type=str, default=HOST_PADRAO,
                        help=f'Endereço do master (default: {HOST_PADRAO})')
    parser.add_argument('--porta', type=int, default=PORTA_PADRAO,
                        help=f'Porta do master (default: {PORTA_PADRAO})')
    args = parser.parse_args()

    executar_worker(host=args.host, porta=args.porta)