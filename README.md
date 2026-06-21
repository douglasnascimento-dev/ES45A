# Simulação de Propagação de Fake News
### Sistemas Paralelos e Distribuídos

Simulação de propagação de fake news em uma população representada por uma matriz bidimensional, implementada em três versões: **Sequencial**, **Paralela** (Multiprocessing) e **Distribuída** (Sockets TCP).

---

## Requisitos

**Python 3.10 ou superior** (testado no 3.14.5)

Instale as dependências antes de rodar qualquer versão:

```bash
pip install psutil matplotlib pillow
```

> Se `pip` não for reconhecido, use `python -m pip install psutil matplotlib pillow`

---

## Arquivos do projeto

```
├── sequencial.py          ← versão sequencial com todas as melhorias
├── paralelo.py            ← versão paralela com multiprocessing
├── master.py              ← versão distribuída (processo coordenador)
├── worker.py              ← versão distribuída (processo de cálculo)
└── rodar_distribuido.py   ← lançador automático da versão distribuída
```

---

## Versão Sequencial

### Como rodar

```bash
python sequencial.py
```

### O que acontece

- Roda benchmark headless (sem animação) com grade `400×400` e `150` gerações
- Ao final, salva a animação em `propagacao.gif`
- Imprime tempo total e distribuição final da população

### Resultado esperado no terminal

```
[MODO HEADLESS] Benchmark CPU (400x400) - 150 gerações
----------------------------------------
 RESULTADOS DO BENCHMARK
----------------------------------------
Tempo Sequencial Puro : X.XXXX segundos
Pico de RAM (alocada) : XX.XX MB
```

### Parâmetros (edite diretamente no final do arquivo)

```python
executar_simulacao(
    linhas=400,
    colunas=400,
    geracoes=150,
    perc_espalhadores=0.02,   # % inicial de espalhadores
    perc_obstaculos=0.08,     # % de obstáculos (células mortas)
    perc_cura=0.002,          # % de fact-checkers iniciais
    semente=15,               # semente do RNG (garante reprodutibilidade)
    modo_grafico=False,       # True = abre janela de animação
)
```

---

## Versão Paralela (Multiprocessing)

### Como rodar

```bash
python paralelo.py
```

> **Windows:** o bloco `if __name__ == "__main__":` no final do arquivo é obrigatório para o multiprocessing funcionar corretamente. Não remova.

### O que acontece

- Detecta automaticamente o número de núcleos da máquina (`os.cpu_count()`)
- Roda benchmark headless com grade `400×400` e `150` gerações
- Ao final, salva a animação em `propagacao_multiprocessing.gif`

### Resultado esperado no terminal

```
[MODO HEADLESS - MULTIPROCESSING] Benchmark CPU (400x400) - 150 gerações | N processos
----------------------------------------
 RESULTADOS DO BENCHMARK
----------------------------------------
Tempo Multiproc. Puro: X.XXXX segundos
Processos Utilizados : N
Pico de RAM (alocada): XX.XX MB
```

### Parâmetros (edite diretamente no final do arquivo)

```python
executar_simulacao(
    linhas=400,
    colunas=400,
    geracoes=150,
    num_processos=N,        # None = detecta automaticamente
    modo_grafico=False,
)
```

---

## Versão Distribuída (Sockets TCP)

A versão distribuída tem **dois processos separados**: o `master` coordena a simulação e os `workers` calculam cada um sua fatia da grade.

### Opção A — Lançador automático (recomendado)

Inicia master e workers automaticamente em um único comando:

```bash
python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80
```

Com animação ao final:

```bash
python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80 --animar
```

Salvando animação em GIF:

```bash
python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80 --salvar propagacao_distribuida.gif
```

---

### Opção B — Terminais separados (para demonstração)

Abra **3 terminais** na mesma pasta do projeto:

**Terminal 1 — Master:**
```bash
python master.py --workers 2 --linhas 100 --colunas 100 --geracoes 80
```

O master vai imprimir:
```
Aguardando 2 worker(s) na porta 65432...
```

**Terminal 2 — Worker 0:**
```bash
python worker.py
```

**Terminal 3 — Worker 1:**
```bash
python worker.py
```

> Os workers se conectam automaticamente ao master. O master os identifica pela **ordem de chegada**.

---

### Resultado esperado no terminal do master

```
=================================================================
  CONFIGURAÇÃO EXPERIMENTAL — AMBIENTE DE EXECUÇÃO (MASTER)
=================================================================
...
Geração 001  [0.052s] | Ign:    8,820 | A:      191 | B:      198 | ...
Geração 002  [0.048s] | Ign:    8,625 | A:      262 | B:      314 | ...
...
==================================================
  RESULTADO FINAL
==================================================
Gerações executadas  : 80
Tempo total          : X.XXXX segundos
Pico de RAM (master) : X.XX MB
Workers utilizados   : 2
==================================================
```

---

### Todos os parâmetros disponíveis

#### `master.py`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--workers` | `2` | Número de workers |
| `--linhas` | `100` | Linhas da grade |
| `--colunas` | `100` | Colunas da grade |
| `--geracoes` | `80` | Máximo de gerações |
| `--espalhadores` | `0.02` | % inicial de espalhadores |
| `--obstaculos` | `0.08` | % de obstáculos |
| `--cura` | `0.002` | % de fact-checkers |
| `--semente` | `15` | Semente RNG |
| `--animar` | desativado | Exibe animação ao final |
| `--salvar` | nenhum | Salva animação (`.gif` ou `.mp4`) |
| `--host` | `0.0.0.0` | Interface de bind |
| `--porta` | `65432` | Porta TCP |

#### `worker.py`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--host` | `localhost` | Endereço do master |
| `--porta` | `65432` | Porta do master |

#### `rodar_distribuido.py`

Aceita todos os parâmetros do master mais `--host` e `--porta` para os workers.

---

## Comparando as três versões (benchmarks)

Para gerar dados comparáveis, rode os mesmos parâmetros nas três versões.

**Exemplo com grade 100×100 e 80 gerações:**

```bash
# 1. Sequencial
#    (edite o __main__ do sequencial.py para linhas=100, colunas=100, geracoes=80)
python sequencial.py

# 2. Paralela
#    (edite o __main__ do paralelo.py para linhas=100, colunas=100, geracoes=150)
python paralelo.py

# 3. Distribuída
python rodar_distribuido.py --workers 2 --linhas 100 --colunas 100 --geracoes 80
```

Anote o tempo de cada execução e calcule o speedup:

```
Speedup = Tempo Sequencial / Tempo Paralela (ou Distribuída)
Eficiência = Speedup / Número de Processos (ou Workers)
```

---

## Solução de problemas

**`ModuleNotFoundError: No module named 'matplotlib'`**
```bash
pip install matplotlib pillow
```

**`ModuleNotFoundError: No module named 'psutil'`**
```bash
pip install psutil
```

**`ConnectionRefusedError` no worker**
O master ainda não iniciou. O worker tenta reconectar automaticamente por 15 segundos. Se persistir, verifique se o master está rodando e se a porta (`65432`) não está em uso por outro programa.

**`ValueError: badly formed help string` (Python 3.14)**
Já corrigido nos arquivos. Se aparecer em outro arquivo, substitua `% texto` por `%% texto` nos argumentos `help=` do argparse.

**Porta 65432 em uso**
Use uma porta diferente:
```bash
python rodar_distribuido.py --workers 2 --porta 5050 --linhas 100 --colunas 100 --geracoes 80
```

---

## Estados e cores da animação

| Cor | Estado |
|---|---|
| Branco | Ignorante (não foi exposto) |
| Vermelho | Afetado pela Fake News A |
| Azul | Afetado pela Fake News B |
| Roxo | Colisão (exposto a A e B) |
| Cinza escuro | Obstáculo (célula morta) |
| Verde | Fact-Checker (agente de cura) |

---

## Melhorias implementadas

| # | Melhoria | Onde |
|---|---|---|
| 1 | `__slots__` na classe Individuo (economiza RAM) | Todas |
| 2 | Múltiplas fake news simultâneas (A e B) | Todas |
| 3 | Mutação da informação (força evolui por geração) | Todas |
| 4 | Relatório automático do ambiente de execução | Todas |
| 5 | Obstáculos / células mortas | Todas |
| 6 | Resistência cruzada entre fake news | Todas |
| 7 | Probabilidade base de adesão (65%) | Todas |
| 8 | Contágio orgânico do fact-checking | Todas |
| 9 | Visualização gráfica com animação | Sequencial / Distribuída |
| 10 | Modo headless para benchmark com medição de RAM | Todas |
| + | RNG determinístico por célula (garante reprodutibilidade) | Todas |
| + | Contagem de estados inline (sem segunda passagem) | Distribuída |
