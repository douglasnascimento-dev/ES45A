"""
Lançador automático — inicia master + workers em subprocessos.

Útil para benchmark: roda tudo com um único comando e coleta
o tempo total ao final.

Uso:
    python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80

Equivalente a abrir manualmente:
    Terminal 1: python master.py --workers 2 --linhas 100 ...
    Terminal 2: python worker.py --id 0
    Terminal 3: python worker.py --id 1
"""

import subprocess
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Lançador distribuído')
    parser.add_argument('--workers',      type=int,   default=2,     help='Número de workers')
    parser.add_argument('--linhas',       type=int,   default=100,   help='Linhas da grade')
    parser.add_argument('--colunas',      type=int,   default=100,   help='Colunas da grade')
    parser.add_argument('--geracoes',     type=int,   default=80,    help='Máximo de gerações')
    parser.add_argument('--espalhadores', type=float, default=0.02,  help='%% espalhadores')
    parser.add_argument('--obstaculos',   type=float, default=0.08,  help='%% obstáculos')
    parser.add_argument('--cura',         type=float, default=0.002, help='%% fact-checkers')
    parser.add_argument('--semente',      type=int,   default=15,    help='Semente RNG')
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
    ]

    # Comandos dos workers
    cmds_workers = [
        [python, 'worker.py']
        for _ in range(args.workers)
    ]

    print(f"Iniciando master + {args.workers} worker(s)...")
    print(f"Grade: {args.linhas}x{args.colunas} | Gerações: {args.geracoes}\n")

    # Inicia master primeiro
    proc_master = subprocess.Popen(cmd_master)
    time.sleep(0.5)  # pequena espera para o master abrir as portas

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