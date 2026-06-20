"""
Lançador automático — inicia master + workers em subprocessos.

Útil para benchmark e teste local: roda tudo com um único comando e
coleta o tempo total ao final.

Uso:
    python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80

    # Porta customizada (útil se a 65432 estiver ocupada):
    python rodar_distribuido.py --workers 4 --porta 5050 --linhas 350

Equivalente a abrir manualmente:
    Terminal 1: python master.py --workers 2 --linhas 100 ...
    Terminal 2: python worker.py
    Terminal 3: python worker.py
"""

import subprocess
import sys
import time
import argparse

HOST_PADRAO  = 'localhost'
PORTA_PADRAO = 65432


def main():
    parser = argparse.ArgumentParser(description='Lançador distribuído')
    parser.add_argument('--workers',      type=int,   default=2,             help='Número de workers')
    parser.add_argument('--linhas',       type=int,   default=100,           help='Linhas da grade')
    parser.add_argument('--colunas',      type=int,   default=100,           help='Colunas da grade')
    parser.add_argument('--geracoes',     type=int,   default=80,            help='Máximo de gerações')
    parser.add_argument('--espalhadores', type=float, default=0.02,          help='%% espalhadores')
    parser.add_argument('--obstaculos',   type=float, default=0.08,          help='%% obstáculos')
    parser.add_argument('--cura',         type=float, default=0.002,         help='%% fact-checkers')
    parser.add_argument('--semente',      type=int,   default=15,            help='Semente RNG')
    parser.add_argument('--animar',       action='store_true',               help='Exibe animação ao final')
    parser.add_argument('--salvar',       type=str,   default=None,          help='Salva animação (.gif ou .mp4)')
    parser.add_argument('--host',         type=str,   default=HOST_PADRAO,
                        help=f'Endereço do master (default: {HOST_PADRAO})')
    parser.add_argument('--porta',        type=int,   default=PORTA_PADRAO,
                        help=f'Porta TCP (default: {PORTA_PADRAO})')
    args = parser.parse_args()

    python = sys.executable

    # Comando do master
    cmd_master = [
        python, 'master.py',
        '--workers',      str(args.workers),
        '--linhas',       str(args.linhas),
        '--colunas',      str(args.colunas),
        '--geracoes',     str(args.geracoes),
        '--espalhadores', str(args.espalhadores),
        '--obstaculos',   str(args.obstaculos),
        '--cura',         str(args.cura),
        '--semente',      str(args.semente),
        '--host',         args.host,
        '--porta',        str(args.porta),
    ]
    if args.animar:
        cmd_master.append('--animar')
    if args.salvar:
        cmd_master.extend(['--salvar', args.salvar])

    # Comandos dos workers
    cmd_worker_base = [
        python, 'worker.py',
        '--host',  args.host,
        '--porta', str(args.porta),
    ]
    cmds_workers = [list(cmd_worker_base) for _ in range(args.workers)]

    print(f"Iniciando master + {args.workers} worker(s)...")
    print(f"Grade: {args.linhas}x{args.colunas} | Gerações: {args.geracoes} | "
          f"Endereço: {args.host}:{args.porta}\n")

    # Inicia master primeiro
    proc_master = subprocess.Popen(cmd_master)

    # Pequena espera para o master abrir o socket de escuta.
    # O worker tem retry de 15 tentativas × 1s, então 1.0s aqui é folgado.
    # NÃO usamos polling com connect: isso consumiria slots do listen queue
    # do master, que pensaria que somos um worker.
    time.sleep(1.0)

    # Inicia workers
    procs_workers = []
    for cmd in cmds_workers:
        procs_workers.append(subprocess.Popen(cmd))
        time.sleep(0.1)

    # Aguarda todos terminarem
    try:
        proc_master.wait()
        for p in procs_workers:
            p.wait()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        proc_master.terminate()
        for p in procs_workers:
            p.terminate()

    print("\nTodos os processos encerrados.")


if __name__ == '__main__':
    main()
