# Relatorio Tecnico
# Implementacao e Avaliacao do Algoritmo ACO para o Problema do Caixeiro Viajante

**Disciplina:** Algoritmos de Otimizacao  
**Autor:** Breno Vambaster  
**Dataset:** DataSet1.csv — 300 cidades, coordenadas (x, y) normalizadas  
**Linguagem:** Python 3.10  

---

## 1. Introducao

### 1.1 Caracterizacao do Problema

O Problema do Caixeiro Viajante (TSP — *Traveling Salesman Problem*) consiste em,
dado um grafo completo ponderado G = (V, E) com |V| = N vertices (cidades) e arestas
com pesos w(i, j) representando a distancia entre cada par de cidades, encontrar o
ciclo hamiltoniano de custo minimo: um percurso que visita cada cidade exatamente
uma vez e retorna a cidade de origem, minimizando a distancia total percorrida.

Formalmente:

```
Minimizar  L(pi) = soma_{k=1}^{N} d(pi_k, pi_{k+1})   onde pi_{N+1} = pi_1

sujeito a: pi e uma permutacao de {1, 2, ..., N}
```

O TSP pertence a classe de problemas NP-dificeis. O espaco de solucoes cresce
fatorialmente com N — para N = 50 cidades ha aproximadamente 3 x 10^64 tours
distintos, tornando a busca exaustiva computacionalmente inviavel. Para N = 5,
o espaco e de apenas 12 tours, mas o objetivo do estudo e o comportamento
do algoritmo conforme N cresce ate 50.

O dataset utilizado contem 300 cidades com coordenadas (x, y) normalizadas no
intervalo [0, ~0.7], armazenadas em `dataset/DataSet1.csv` (uma cidade por linha,
sem cabecalho). As distancias entre cidades sao calculadas como distancias
euclidianas sobre esse espaco normalizado.

### 1.2 Justificativa da Abordagem

A Otimizacao por Colonia de Formigas (ACO) e uma metaheuristica de inteligencia
coletiva inspirada no comportamento de forrageamento de formigas reais. Formigas
reais depositam feromonio nos caminhos percorridos; formigas subsequentes tendem
a seguir trilhas com maior concentracao de feromonio, criando um mecanismo de
aprendizado coletivo que converge para caminhos curtos.

O ACO e naturalmente adequado ao TSP por tres razoes principais:

1. O TSP e representado como grafo — feromonio pode ser modelado diretamente
   nas arestas.
2. A construcao incremental de solucoes (cidade por cidade) e analogica ao
   caminho percorrido por uma formiga.
3. O balanco entre exploracao e explotacao e controlavel pelos parametros
   alfa (feromonio) e beta (heuristica).

---

## 2. Materiais e Metodos

### 2.1 Visao Geral do Algoritmo

O algoritmo implementado e o **Ant System (AS)** classico com **selecao por
Torneio**, conforme especificado no enunciado do trabalho. Os componentes
principais sao:

| Componente | Descricao |
|---|---|
| Matriz de feromonio t(i,j) | Intensidade aprendida em cada aresta |
| Matriz heuristica n(i,j) | Informacao a priori: n(i,j) = 1 / d(i,j) |
| Regra de selecao | Torneio com tamanho configuravel |
| Atualizacao de feromonio | AS classico: evaporacao global + deposito Q/L |
| Criterio de parada | Numero fixo de iteracoes OU estagnacao (early stopping) |

### 2.2 Parametros do Algoritmo

| Parametro | Simbolo | Descricao |
|---|---|---|
| Numero de formigas | m | Formigas por iteracao |
| Numero de iteracoes | T | Iteracoes maximas |
| Expoente de feromonio | alfa | Peso da informacao aprendida |
| Expoente heuristico | beta | Peso da distancia imediata |
| Taxa de evaporacao | rho | Fracao de feromonio evaporado por iteracao |
| Coeficiente de deposito | Q | Escala do deposito: deposito = Q / L |
| Feromonio inicial | tau_0 | Valor inicial em todas as arestas |
| Tamanho do torneio | k | Candidatos por passo de selecao |

### 2.3 Pseudocodigo Completo

```
ENTRADA: matriz de distancias D[N][N], parametros m, T, alfa, beta, rho, Q, tau_0, k

INICIALIZACAO:
  tau[i][j] <- tau_0  para todo (i,j)
  eta[i][j] <- 1 / D[i][j]  para todo i != j
  melhor_global <- infinito
  melhor_tour_global <- nulo

LACO PRINCIPAL (t = 1 ate T):

  PARA cada formiga a = 1 ate m:
    cidade_atual <- cidade aleatoria em {0..N-1}
    tour_a <- [cidade_atual]
    nao_visitadas <- {0..N-1} \ {cidade_atual}

    ENQUANTO nao_visitadas nao for vazio:
      # -- SELECAO POR TORNEIO --
      candidatos <- amostra aleatoria de min(k, |nao_visitadas|) cidades
      melhor_cand <- argmax_{j em candidatos} [ tau[cidade_atual][j]^alfa * eta[cidade_atual][j]^beta ]
      tour_a <- tour_a + [melhor_cand]
      nao_visitadas <- nao_visitadas \ {melhor_cand}
      cidade_atual <- melhor_cand

    L_a <- soma das distancias do tour_a (fechado: ultima -> primeira)

    SE L_a < melhor_global:
      melhor_global <- L_a
      melhor_tour_global <- tour_a

  # -- ATUALIZACAO DE FEROMONIO (AS classico) --
  # 1. Evaporacao global
  tau[i][j] <- (1 - rho) * tau[i][j]  para todo (i,j)

  # 2. Deposito por cada formiga
  PARA cada formiga a:
    delta <- Q / L_a
    PARA cada aresta (i,j) em tour_a:
      tau[i][j] <- tau[i][j] + delta
      tau[j][i] <- tau[j][i] + delta   # grafo simetrico

  # -- CRITERIO DE PARADA ANTECIPADA (se habilitado) --
  SE melhor_global nao melhorou nas ultimas X iteracoes:
    INTERROMPER

SAIDA: melhor_tour_global, melhor_global
```

### 2.4 Selecao por Torneio

Na selecao por Torneio, em vez de calcular probabilidades para todos os candidatos
e sortear (como na roleta), sorteia-se um subconjunto de **k candidatos** e
seleciona-se deterministicamente aquele com maior atratividade:

```
atratividade(j) = tau(i,j)^alfa * eta(i,j)^beta
```

Propriedades:
- k = 1: selecao aleatoria pura (sem informacao)
- k = N: sempre escolhe o melhor vizinho (guloso)
- k = 2: balanco entre aleatoriedade e direcao (valor padrao neste trabalho)

Em comparacao com a roleta, o torneio e mais simples de implementar, nao requer
normalizacao de probabilidades e oferece controle intuitivo sobre o nivel de
greediness via k.

### 2.5 Atualizacao de Feromonio — AS Classico

A atualizacao segue a formula do Ant System original (Dorigo et al., 1992):

```
Evaporacao: tau(i,j) <- (1 - rho) * tau(i,j)     para todo (i,j)

Deposito:   delta_k  = Q / L_k
            tau(i,j) <- tau(i,j) + delta_k         para toda aresta (i,j) no tour k
```

O parametro Q escala a magnitude do deposito. Como delta = Q/L, solucoes de menor
distancia depositam automaticamente mais feromonio — esse e o mecanismo de
aprendizado coletivo do algoritmo.

### 2.6 Criterio de Parada Antecipada (Early Stopping)

No experimento final, o algoritmo monitora quantas iteracoes consecutivas passam
sem melhoria no melhor resultado global. Se esse contador atingir o limiar X = 30,
o laco encerra antes do maximo de 500 iteracoes. Isso economiza tempo quando o
algoritmo ja convergiu e evita iteracoes estereis.

### 2.7 Implementacao

**Arquivo principal:** `aco.py` — classe `ACO`

Decisoes de implementacao relevantes:

- **Matriz de distancias pre-computada (numpy):** distancias euclidianas calculadas
  uma unica vez antes do laco principal. Uso de broadcasting vetorizado:
  `distances = sqrt(sum((coords[i] - coords[j])^2))`.

- **Matriz heuristica pre-computada:** `eta[i][j] = 1/d[i][j]` calculada no
  construtor. Evita divisoes repetidas dentro do laco mais interno.

- **Simetria do feromonio:** tau e uma matriz NxN simetrica. O deposito e aplicado
  em ambas as direcoes (i->j e j->i) simultaneamente, correto para o TSP simetrico.

- **Feromonio inicial:** todas as arestas iniciam com tau_0 = 0.1, fornecendo
  exploracao uniforme nas primeiras iteracoes.

### 2.8 Configuracao dos Experimentos

#### Experimento A — Verificacao de Funcionamento

```
N = 5 cidades       n_ants = 10     n_iter = 30
alfa = 1.0          beta = 1.0      rho = 0.03
Q = 10              tau0 = 0.1      k_torneio = 2
```

Uma unica execucao. Plota grafico de convergencia para verificar que o algoritmo
encontra e mantem a melhor solucao ao longo das iteracoes.

#### Experimento B — Influencia de Alfa e Beta

Mesmos parametros do A, variando apenas o par (alfa, beta):

| Configuracao | alfa | beta |
|---|---|---|
| Config 1 | 0.6 | 0.2 |
| Config 2 | 0.2 | 0.6 |

10 execucoes independentes por configuracao. Comparacao via media, desvio padrao
e boxplot com pontos individuais.

#### Experimento C — Influencia da Taxa de Evaporacao

Melhores alfa e beta do experimento B. Variacoes de rho:

| rho | Comportamento esperado |
|---|---|
| 0.01 | Evaporacao muito lenta — feromonio acumula, menor diversidade |
| 0.05 | Evaporacao moderada-baixa |
| 0.10 | Evaporacao moderada |
| 0.20 | Evaporacao rapida — maior diversidade, convergencia mais lenta |

10 execucoes por taxa. Comparacao via media, desvio padrao e boxplot.

#### Experimento Final — Versao Final em Escala

Parametros otimizados dos experimentos B e C. Instancias maiores:

```
N = 10, 20, 50 cidades
n_ants = 20     n_iter_max = 500    early_stopping = 30
Q = 10          tau0 = 0.1          k_torneio = 2
```

10 execucoes por tamanho. Metricas: media, mediana, moda, desvio padrao e tempo
medio de execucao.

---

## 3. Resultados e Discussao

### 3.1 Experimento A — Verificacao de Funcionamento

**Configuracao:** 5 cidades, alfa=1, beta=1, rho=0.03, Q=10, tau0=0.1, 30 iteracoes.

A execucao encontrou a melhor rota com distancia **1.2431** ja nas primeiras
iteracoes e a manteve ate o final.

O grafico de convergencia (plots/A_convergencia.png) apresenta duas curvas:

- **Linha azul (media das formigas por iteracao):** oscila entre ~1.27 e ~1.49.
  Com apenas 5 cidades e 10 formigas partindo de pontos aleatorios, a media
  reflete a diversidade de exploracoes — algumas formigas encontram rotas proximas
  do otimo, outras exploram rotas mais longas.

- **Linha vermelha (melhor acumulado):** permanece plana em 1.2431 desde a
  segunda iteracao. Isso demonstra que o algoritmo encontra a solucao otima muito
  rapidamente para 5 cidades e a mantem ao longo de todas as 30 iteracoes.

**Conclusao:** o algoritmo esta funcionando corretamente. A linha vermelha
monotonicamente nao-crescente confirma que o feromonio esta sendo depositado e
a aprendizagem ocorre. A oscilacao da media e esperada — as formigas continuam
explorando, mas a melhor solucao ja foi encontrada.

---

### 3.2 Experimento B — Influencia de Alfa e Beta

**Configuracao:** 5 cidades, rho=0.03, Q=10, tau0=0.1, 10 runs por config.

| Configuracao | Media | Mediana | Moda | Desvio Padrao |
|---|---|---|---|---|
| alfa=0.6, beta=0.2 | 1.2443 | 1.2431 | 1.2431 | 0.0035 |
| alfa=0.2, beta=0.6 | 1.2431 | 1.2431 | 1.2431 | 0.0000 |

**Melhor configuracao selecionada: alfa=0.2, beta=0.6**

O boxplot (plots/B_alpha_beta_boxplot.png) mostra que:

- **alfa=0.6, beta=0.2** apresenta um outlier em ~1.2549 e desvio padrao de 0.0035,
  indicando que em algumas execucoes o algoritmo falha em convergir ao otimo.
  Com peso alto em alfa (feromonio) e baixo em beta (heuristica), o algoritmo
  depende mais do feromonio aprendido. Com apenas 30 iteracoes e rho=0.03
  (evaporacao lenta), o feromonio acumulado pode nao refletir a qualidade das
  rotas corretamente nas primeiras iteracoes, levando ocasionalmente a rotas
  subotimas.

- **alfa=0.2, beta=0.6** apresenta desvio padrao zero — todas as 10 execucoes
  convergiram ao mesmo valor otimo. Com peso maior em beta, a heuristica de
  distancia (escolher vizinhos proximos) guia a busca com mais eficacia em instancias
  pequenas, onde a informacao greedy e suficiente para encontrar o otimo.

**Interpretacao:** para instancias pequenas (5 cidades), dar mais peso a heuristica
(beta alto) e mais robusto do que confiar no feromonio aprendido. Isso e coerente
com a literatura: em instancias pequenas, a informacao greedy e suficiente; em
instancias maiores, o feromonio se torna mais importante.

---

### 3.3 Experimento C — Influencia da Taxa de Evaporacao

**Configuracao:** 5 cidades, alfa=0.2, beta=0.6, Q=10, tau0=0.1, 10 runs por taxa.

| rho | Media | Mediana | Moda | Desvio Padrao |
|---|---|---|---|---|
| 0.01 | 1.2431 | 1.2431 | 1.2431 | 0.0000 |
| 0.05 | 1.2431 | 1.2431 | 1.2431 | 0.0000 |
| 0.10 | 1.2431 | 1.2431 | 1.2431 | 0.0000 |
| 0.20 | 1.2431 | 1.2431 | 1.2431 | 0.0000 |

**Melhor taxa selecionada: rho=0.01** (empate — primeira na lista)

O boxplot (plots/C_evaporacao_boxplot.png) mostra que todas as quatro taxas
produzem o mesmo resultado otimo em todas as 10 execucoes, sem variancia.

**Interpretacao:** com 5 cidades e os parametros alfa=0.2, beta=0.6 (heuristica
dominante), o problema e resolvido otimamente independentemente da taxa de
evaporacao. A combinacao de beta alto com o pequeno espaco de busca garante
convergencia ao otimo em poucas iteracoes, antes que a evaporacao tenha impacto
significativo.

Esse resultado e esperado do ponto de vista cientifico: a influencia da taxa de
evaporacao so se torna significativa em instancias maiores, onde a diversidade
de exploracoes e a persistencia do feromonio ao longo de muitas iteracoes afetam
a qualidade da solucao. Nos experimentos finais (50 cidades), esse efeito seria
mais pronunciado.

---

### 3.4 Experimento Final — Versao Final em Escala

**Configuracao final:** alfa=0.2, beta=0.6, rho=0.01, Q=10, tau0=0.1,
n_ants=20, n_iter_max=500, early_stopping=30.

#### 10 cidades

| Metrica | Valor |
|---|---|
| Media | ~2.01 |
| Mediana | ~2.00 |
| Moda | ~1.999 |
| Desvio padrao | ~0.02 |
| Tempo medio | ~0.035 s |

O algoritmo converge rapidamente (em geral antes de 80 iteracoes, via early
stopping). O desvio padrao baixo indica alta consistencia entre execucoes.
O grafico Final_10.png mostra a linha vermelha (melhor acumulado) descendo em
poucos degraus nas primeiras iteracoes e estabilizando.

#### 20 cidades

| Metrica | Valor |
|---|---|
| Media | ~3.69 |
| Mediana | ~3.73 |
| Moda | ~3.76 |
| Desvio padrao | ~0.10 |
| Tempo medio | ~0.07 s |

O desvio padrao aumenta em relacao a 10 cidades, refletindo maior variabilidade
do algoritmo em espacos de busca maiores. O grafico Final_20.png mostra a linha
vermelha descendo abruptamente nas primeiras 2-3 iteracoes e depois de forma mais
gradual, convergindo em ~30-65 iteracoes.

#### 50 cidades

| Metrica | Valor |
|---|---|
| Media | ~10.85 |
| Mediana | ~10.94 |
| Moda | ~11.01 |
| Desvio padrao | ~0.37 |
| Tempo medio | ~0.18 s |

Com 50 cidades, a variabilidade entre execucoes e maior (desvio ~0.37). O grafico
Final_50.png mostra uma convergencia progressiva com varios degraus de melhoria
distribuidos ao longo de 50-80 iteracoes antes do early stopping. Isso indica
que o feromonio aprendido esta contribuindo para melhorias ao longo do tempo —
o algoritmo nao converge instantaneamente como nos casos de 5 cidades.

**Escala de tempo:** o tempo de execucao segue a complexidade teorica O(m * T * N^2):

| N | Tempo (s) | Razao (relativo a N=10) |
|---|---|---|
| 10 | ~0.035 | 1.0x |
| 20 | ~0.070 | 2.0x |
| 50 | ~0.180 | 5.1x |

O crescimento e aproximadamente linear em N para este intervalo, pois o early
stopping encerra antes das 500 iteracoes maximas — o numero real de iteracoes
tambem cresce com N.

---

### 3.5 Consideracoes Gerais

**Por que os experimentos B e C mostram pouca diferenca com 5 cidades?**
Com apenas 5 cidades (12 tours possiveis), qualquer variante do ACO encontra o
otimo global rapidamente. A diferenciacao entre configuracoes de parametros so se
manifesta em instancias maiores, onde o espaco de busca e suficientemente grande
para que o feromonio e a taxa de evaporacao influenciem o caminho de convergencia.

**Comportamento do early stopping:**
Com N=50, o algoritmo encerrou em media ao redor de 60-80 iteracoes (de 500
maximas), economizando ~85% do tempo computacional sem perda de qualidade. Isso
confirma a utilidade do criterio de parada antecipada.

**Selecao por Torneio:**
A implementacao com k=2 (torneio binario) mostrou resultados consistentes. A
selecao e mais rapida que a roleta (O(k) por passo versus O(N)) e oferece controle
intuitivo sobre o greediness. Para trabalhos futuros, seria interessante testar
k=3 e k=5 em instancias maiores.

---

## 4. Estrutura do Codigo

```
TSP_ACO/
├── aco.py            # Classe ACO: selecao por torneio, AS classico, early stopping
├── experiments.py    # Experimentos A/B/C/Final, estatisticas, plots
├── show.py           # Visualizacao matplotlib: tour + convergencia
└── show_map.py       # Visualizacao interativa via folium (HTML + OpenStreetMap)
```

### Classe ACO (aco.py) — metodos principais

| Metodo | Funcao |
|---|---|
| `__init__` | Inicializa tau, eta, parametros |
| `_select_next(current, unvisited)` | Selecao por torneio |
| `_build_tour()` | Constroi um tour completo para uma formiga |
| `_tour_distance(tour)` | Calcula distancia total do tour (fechado) |
| `_update_pheromones(tours, dists)` | Evaporacao + deposito AS |
| `run()` | Laco principal, retorna dicionario com resultados |

### Visualizacao (show_map.py)

O `show_map.py` gera um mapa HTML interativo usando a biblioteca **folium**:
- Tiles OpenStreetMap como caminho de fundo
- Polyline azul representando o tour otimo
- Marcador verde (inicio) e vermelho (fim) diferenciados
- Popup em cada cidade com indice e posicao na rota
- Painel de resumo com distancia e parametros usados

As coordenadas abstratas do DataSet1.csv sao mapeadas linearmente para uma
regiao geografica configuravel no topo do arquivo (parametros LAT_MIN/MAX
e LON_MIN/MAX).

---

## 5. Pre-requisitos e Execucao

```bash
pip install numpy matplotlib folium

# Rodar todos os experimentos (gera plots/ automaticamente)
python experiments.py

# Visualizacao matplotlib (salva plots/show_tour.png)
python show.py

# Mapa interativo (abre tour_map.html no navegador)
python show_map.py
```

---

## 6. Conclusao

O algoritmo ACO com selecao por Torneio foi implementado com sucesso e avaliado
em instancias de 5 a 50 cidades do dataset DataSet1.csv.

Os principais achados sao:

1. **O algoritmo converge corretamente** — confirmado pelo grafico de convergencia
   do experimento A, onde a melhor solucao acumulada decresce monotonicamente.

2. **Beta alto (peso heuristico) e mais robusto para instancias pequenas** — o par
   alfa=0.2, beta=0.6 mostrou desvio padrao zero em 10 execucoes no experimento B,
   enquanto alfa=0.6, beta=0.2 apresentou outliers.

3. **A taxa de evaporacao nao influencia resultados em 5 cidades** — todas as
   quatro taxas testadas convergiram ao otimo. Seu efeito seria mais pronunciado
   em instancias maiores.

4. **Early stopping e eficaz** — encerra em media 80% antes do limite de iteracoes
   para N=50, sem perda de qualidade da solucao.

5. **Escalabilidade razoavel** — o tempo cresce de forma controlada com N,
   permanecendo abaixo de 0.2 s para 50 cidades com early stopping.

Como trabalhos futuros, sugere-se: (a) testar tamanhos de torneio k > 2 em
instancias maiores; (b) comparar com a selecao por roleta nas mesmas instancias;
(c) avaliar o impacto da taxa de evaporacao em N >= 100 cidades, onde o efeito
deve se tornar estatisticamente significativo.
