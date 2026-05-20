# Relatório Técnico: Solução do Problema do Caixeiro Viajante via Otimização por Colônia de Formigas

**Projeto:** TSP_ACO  
**Autor:** Breno Vambaster  
**Dataset:** 225 cidades do subcontinente indiano  

---

## Resumo

Este relatório descreve a implementação e comparação de três variantes do algoritmo de Otimização por Colônia de Formigas (ACO) aplicadas ao Problema do Caixeiro Viajante (TSP). São implementadas as variantes ACS (Ant Colony System), Elitista e MaxMin, avaliadas experimentalmente em instâncias de tamanho crescente. O projeto também inclui visualização geográfica da rota ótima sobre um mapa real com 225 cidades.

---

## 1. Introdução

### 1.1 O Problema do Caixeiro Viajante

O Problema do Caixeiro Viajante (TSP — *Traveling Salesman Problem*) é um dos problemas de otimização combinatória mais estudados na ciência da computação. Formalmente, dado um grafo completo ponderado `G = (V, E)` com `|V| = N` vértices (cidades) e arestas com pesos `w(i, j)` (distâncias), busca-se encontrar o ciclo hamiltoniano de custo mínimo — isto é, um percurso que visita cada cidade exatamente uma vez e retorna à origem, minimizando a distância total percorrida.

O TSP pertence à classe de problemas NP-difíceis. Para instâncias com `N = 225` cidades, o espaço de soluções possui `(N-1)!/2 ≈ 10^450` rotas distintas, tornando a busca exaustiva computacionalmente inviável. Isso motiva o uso de metaheurísticas.

### 1.2 Justificativa da Abordagem ACO

A Otimização por Colônia de Formigas é uma metaheurística de inteligência coletiva (*swarm intelligence*) inspirada no comportamento de formigas reais ao buscar alimento. Formigas reais depositam feromônio nos caminhos percorridos; formigas subsequentes tendem a seguir trilhas com maior concentração de feromônio, criando um mecanismo de aprendizado coletivo que converge para caminhos curtos.

O ACO é particularmente adequado ao TSP porque:

1. O TSP possui estrutura de grafo sobre a qual feromônio pode ser modelado nas arestas.
2. A construção incremental de soluções (cidade por cidade) se encaixa naturalmente na metáfora da formiga caminhando.
3. O equilíbrio entre exploração e exploração (*exploration vs. exploitation*) pode ser controlado pelos parâmetros α e β.

---

## 2. Dataset

### 2.1 Descrição

O dataset contém **225 cidades** localizadas no noroeste e centro do subcontinente indiano. Para cada cidade estão disponíveis:

| Arquivo | Conteúdo | Formato |
|---|---|---|
| `location_ll.txt` | Coordenadas geográficas (latitude, longitude) | Tab-separated, 225 linhas |
| `names.txt` | Nome de cada cidade | Uma por linha |
| `distance.txt` | Matriz de distâncias euclidianas | 225×225 |
| `road_distance.txt` | Matriz de distâncias por estrada | 225×225 |
| `travel_time.txt` | Matriz de tempos de viagem | 225×225 |

### 2.2 Pré-processamento

Em `show.py`, as coordenadas geográficas passam por normalização antes de serem usadas pelo algoritmo:

```python
scaler = MinMaxScaler(feature_range=(-90, 175))
location[:, 1] = scaler.fit_transform(location[:, 1].reshape(-1, 1))  # longitude

scaler = MinMaxScaler(feature_range=(-80, 75))
location[:, 0] = scaler.fit_transform(location[:, 0].reshape(-1, 1))  # latitude
```

**Decisão de projeto:** os intervalos de escala foram escolhidos para corresponder ao espaço de coordenadas do mapa de fundo (`assets/map.png`), permitindo que os pontos das cidades sejam desenhados com precisão sobre o mapa usando o sistema de coordenadas da janela turtle (`setworldcoordinates(-180, -90, 180, 90)`).

As distâncias entre cidades são calculadas como **distâncias euclidianas** sobre as coordenadas normalizadas, não como distâncias geodésicas reais. Para os fins comparativos deste trabalho, essa simplificação é aceitável.

---

## 3. Fundamentação Teórica

### 3.1 Modelo ACO

O estado do sistema ACO é definido pelos **feromônios** nas arestas do grafo: `τ(i, j)` representa a intensidade de feromônio na aresta entre as cidades `i` e `j`. Cada iteração consiste em:

1. Cada formiga constrói uma solução completa (tour) partindo de uma cidade aleatória.
2. Feromônio é depositado nas arestas utilizadas.
3. Feromônio evapora em todas as arestas.

### 3.2 Regra de Seleção Estocástica

A formiga na cidade `i`, escolhendo a próxima cidade `j` dentre as não visitadas `U`, aplica a seguinte regra de roleta ponderada:

```
P(j | i) = [τ(i,j)^α · η(i,j)^β] / Σ_{k ∈ U} [τ(i,k)^α · η(i,k)^β]
```

Onde:
- `τ(i, j)` — feromônio na aresta (i, j)
- `η(i, j)` — heurística de visibilidade (inversamente proporcional à distância)
- `α` — expoente que controla a influência do feromônio
- `β` — expoente que controla a influência da heurística

**Implementação da heurística:** a implementação adota uma versão normalizada da visibilidade:

```python
heuristic_total = Σ d(current, k) para todo k não visitado

η(current, j) = heuristic_total / d(current, j)
```

Isso é equivalente a `1/d(current, j)` em termos de ordenação relativa das probabilidades (a soma `heuristic_total` é constante para uma dada posição), mas fornece melhor estabilidade numérica ao evitar valores muito pequenos quando as distâncias são grandes.

---

## 4. Variantes do Algoritmo

### 4.1 ACS — Ant Colony System (`_acs`)

**Estratégia:** depósito de feromônio constante por formiga, seguido de evaporação global.

**Pseudocódigo:**
```
para cada iteração t = 1..T:
    para cada formiga k = 1..m:
        construir tour via seleção estocástica
        depositar feromônio: τ(i,j) += W / distância_k  para toda aresta (i,j) no tour
        atualizar melhor solução global se distância_k < melhor_global
    evaporar: τ(i,j) *= (1 - ρ)  para toda aresta (i,j)
```

**Decisão de projeto:** o depósito é feito *durante* a iteração (após cada formiga), enquanto a evaporação ocorre *após* todas as formigas. Isso cria um viés sutil: formigas que constroem tours no início da iteração recebem feromônio depositado pelas formigas anteriores antes da evaporação — um efeito de aprendizado incremental dentro da própria iteração.

**Características:** explora o espaço de maneira mais uniforme. Adequado quando não há boas soluções conhecidas de antemão.

---

### 4.2 Elitista (`_elitist`)

**Estratégia:** idêntica ao ACS, com adição de um depósito bônus na melhor rota global encontrada até o momento.

**Pseudocódigo:**
```
para cada iteração t = 1..T:
    para cada formiga k = 1..m:
        construir tour
        depositar feromônio normal
        atualizar melhor_global se necessário
    depositar feromônio extra na melhor_global:
        τ(i,j) += elitist_weight * (W / distância_global)
    evaporar: τ(i,j) *= (1 - ρ)
```

**Decisão de projeto:** o parâmetro `elitist_weight` controla o peso relativo do depósito elitista em relação ao depósito regular. Com `elitist_weight = 1.0`, a melhor formiga contribui com o mesmo peso de uma formiga comum, mas como o depósito é proporcional a `1/distância`, soluções melhores (menor distância) recebem automaticamente mais feromônio — isso é intensional.

**Características:** favorece convergência mais rápida para soluções de alta qualidade. Risco de convergência prematura em ótimos locais se `elitist_weight` for muito alto ou `rho` muito baixo.

---

### 4.3 MaxMin (`_max_min`)

**Estratégia:** feromônio limitado por limites dinâmicos `[τ_min, τ_max]`, com dois regimes de depósito.

**Fase 1 — Primeiras 75% das iterações (exploração):**
```
usar melhor da iteração atual para depósito
τ_max = W / distância_iteração_melhor
```

**Fase 2 — Últimas 25% das iterações (refinamento):**
```
usar melhor global para depósito
τ_max = W / distância_global_melhor
```

**Em ambas as fases:**
```
τ_min = τ_max × min_scaling_factor
para toda aresta (i,j):
    τ(i,j) *= (1 - ρ)
    τ(i,j) = max(τ_min, min(τ_max, τ(i,j)))
```

**Decisão de projeto — dois regimes:** A transição em 75% é uma heurística empírica amplamente usada na literatura de MaxMin. Nos primeiros 75%, o algoritmo usa a melhor solução da iteração atual (não o global), forçando mais diversidade — formigas diferentes encontram ótimos locais distintos. Nos últimos 25%, concentra o depósito no global best, afunilando a busca para refinar a melhor solução encontrada.

**Decisão de projeto — limites dinâmicos:** `τ_max` é recalculado a cada iteração com base na qualidade atual das soluções. Isso é diferente das implementações originais do MaxMin-AS (onde τ_max é fixo). A abordagem dinâmica se adapta automaticamente à escala do problema.

**Características:** melhor balanço exploração/exploitação entre as três variantes. Tende a produzir melhores soluções em instâncias maiores ao custo de maior complexidade de ajuste de parâmetros.

---

## 5. Implementação

### 5.1 Estrutura de Dados: Classe `Edge`

```python
class Edge:
    def __init__(self, a, b, weight, initial_pheromone):
        self.a = a           # índice da cidade de origem
        self.b = b           # índice da cidade de destino
        self.weight = weight # distância euclidiana
        self.pheromone = initial_pheromone
```

**Decisão de projeto:** arestas com peso zero recebem `weight = 1e-10` para evitar divisão por zero na heurística. Na prática, isso não ocorre com cidades em posições distintas.

A matriz de adjacência `edges[N][N]` é simétrica: `edges[i][j] = edges[j][i]` (mesmo objeto). Isso garante que depósitos de feromônio em qualquer direção afetem a mesma aresta, reduzindo consumo de memória pela metade.

### 5.2 Representação do Grafo

```python
self.edges = [[None] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        self.edges[i][j] = self.edges[j][i] = Edge(i, j, dist(i, j), initial_pheromone)
```

Para `N = 225`, isso cria `225 × 224 / 2 = 25.200` objetos `Edge`. A complexidade espacial é `O(N²)`.

**Decisão de projeto:** a distância é calculada como distância euclidiana diretamente das coordenadas fornecidas em `nodes`, sem utilizar as matrizes pré-computadas do dataset (`distance.txt`). Isso torna o módulo `aco_tsp.py` auto-contido e independente do formato do dataset.

### 5.3 Classe `Ant`

Cada formiga mantém:
- `tour`: lista ordenada de índices de cidades visitadas
- `distance`: distância total do tour atual

O método `find_tour()` constrói um tour completo:

```python
def find_tour(self):
    self.tour = [random.randint(0, N - 1)]  # cidade inicial aleatória
    while len(self.tour) < N:
        self.tour.append(self._select_node())
    return self.tour
```

**Decisão de projeto:** a cidade inicial é selecionada aleatoriamente a cada chamada. Isso aumenta a diversidade da busca, pois formigas exploram o grafo a partir de diferentes pontos de partida.

### 5.4 Seleção por Roleta Ponderada

O algoritmo de seleção percorre a lista de nós não visitados duas vezes:

1. **Primeira passagem:** calcula o denominador `roulette_wheel` (soma de todos os pesos).
2. **Segunda passagem:** percorre os nós acumulando pesos até superar `random_value`.

```python
random_value = random.uniform(0.0, roulette_wheel)
wheel_position = 0.0
for node in unvisited_nodes:
    wheel_position += peso(node)
    if wheel_position >= random_value:
        return node
```

**Complexidade:** `O(N)` por seleção, `O(N²)` para construir um tour completo. Com `m` formigas e `T` iterações: `O(m · T · N²)` no total.

**Observação de implementação:** a lista `unvisited_nodes` é recriada a cada chamada a `_select_node()` via list comprehension `[n for n in range(N) if n not in self.tour]`. A operação `in` sobre uma lista tem complexidade `O(N)`, tornando a criação da lista `O(N²)`. Para instâncias muito grandes (`N > 1000`), converter `self.tour` para um `set` reduziria isso para `O(N)`.

### 5.5 Gestão de Feromônios

O método `_add_pheromone` é compartilhado entre as três variantes:

```python
def _add_pheromone(self, tour, distance, weight=1.0):
    deposit = (pheromone_deposit_weight / distance) * weight
    for i in range(N):
        edges[tour[i]][tour[(i+1) % N]].pheromone += deposit
```

O operador `(i+1) % N` fecha o ciclo hamiltoniano, adicionando feromônio na aresta que retorna da última cidade à primeira.

---

## 6. Visualização

`show.py` implementa a visualização usando o módulo `turtle` da biblioteca padrão do Python. A janela é configurada com coordenadas do mundo reais:

```python
screen.setworldcoordinates(-180, -90, 180, 90)
```

Isso permite usar as coordenadas geográficas normalizadas diretamente como coordenadas de tela, sem conversão adicional.

**Codificação de cores:**
- Verde (dot de 30px): cidade de partida
- Azul (dot de 5px): cidades intermediárias
- Vermelho (dot de 30px): última cidade antes do retorno

**Decisão de projeto:** a formiga percorre `route[0]` → `route[1]` → ... → `route[-1]` → retorno implícito a `route[0]`. O retorno não é desenhado explicitamente na visualização — isso é intencional para destacar visualmente o início/fim da rota.

---

## 7. Setup Experimental e Análise de Resultados

### 7.1 Configuração do Benchmark

O script `aco_tsp.py` (quando executado diretamente) realiza:

- **Tamanhos de instância:** 10, 20, ..., 100 cidades (iterações 1 a 10)
- **Repetições por tamanho:** 20 trials com nós aleatórios em `[-400, 400]²`
- **Parâmetros fixos:** `colony_size=5`, `steps=50`
- **Saída:** arquivo `out.csv`

### 7.2 Análise dos Resultados

A análise do arquivo `result.csv` (benchmark pré-computado) permite extrair as seguintes observações:

**Tempo de execução** escala aproximadamente com `O(N²)` para `N` crescente:

| Tamanho (cidades) | Tempo médio (s) |
|---|---|
| 10 | ~0.021 |
| 20 | ~0.080 |
| 30–100 | cresce quadraticamente |

Isso é coerente com a complexidade teórica `O(m · T · N²)`.

**Qualidade da solução:** para instâncias pequenas (10 cidades), as três variantes produzem resultados idênticos ou muito próximos — o espaço de busca é suficientemente pequeno para que todas convirjam para o mesmo ótimo local. A diferenciação entre as variantes fica mais pronunciada para instâncias maiores.

---

## 8. Como Executar

### 8.1 Pré-requisitos

```bash
pip install numpy matplotlib scikit-learn tqdm
```

Python 3.8 ou superior. O módulo `tkinter` (para `turtle`) é incluído na maioria das distribuições Python padrão.

### 8.2 Execução da Visualização

```bash
python show.py
```

Carrega `tsp dataset/location_ll.txt`, normaliza as coordenadas, executa ACS com 15 formigas e 50 iterações, imprime o resultado no console e abre a janela de visualização.

### 8.3 Execução do Benchmark

```bash
python aco_tsp.py
```

Gera `out.csv` com métricas de tempo e distância para as três variantes em instâncias de 10 a 100 cidades.

### 8.4 Uso Programático

```python
from aco_tsp import SolveTSPUsingACO

nodes = [(lat, lon), ...]   # lista de coordenadas (qualquer escala)

# ACS
model = SolveTSPUsingACO(mode='ACS', colony_size=15, steps=100, nodes=nodes)
runtime, distance = model.run()
model.plot(save=True, name='acs_result.png')

# Elitista
model = SolveTSPUsingACO(mode='Elitist', colony_size=15, steps=100,
                          elitist_weight=1.5, nodes=nodes)
runtime, distance = model.run()

# MaxMin
model = SolveTSPUsingACO(mode='MaxMin', colony_size=15, steps=100,
                          min_scaling_factor=0.001, nodes=nodes)
runtime, distance = model.run()
```

### 8.5 Referência Completa de Parâmetros

| Parâmetro | Padrão | Descrição | Efeito de aumentar |
|---|---|---|---|
| `colony_size` | 10 | Formigas por iteração | Mais exploração, mais lento |
| `steps` | 100 | Número de iterações | Melhor qualidade, mais lento |
| `alpha` | 1.0 | Peso do feromônio | Mais exploitação de trilhas aprendidas |
| `beta` | 3.0 | Peso da heurística | Mais guloso (prefere vizinhos próximos) |
| `rho` | 0.1 | Taxa de evaporação | Esquecimento mais rápido, mais diversidade |
| `pheromone_deposit_weight` | 1.0 | Escala do depósito | Intensifica sinal de feromônio |
| `initial_pheromone` | 1.0 | Feromônio inicial | Influencia exploração inicial |
| `elitist_weight` | 1.0 | Bônus elitista *(Elitist)* | Convergência mais rápida |
| `min_scaling_factor` | 0.001 | Razão τ_min/τ_max *(MaxMin)* | Aumenta diversidade mínima |

---

## 9. Estrutura do Código

```
TSP_ACO/
├── aco_tsp.py          # Implementação central do ACO
│   ├── SolveTSPUsingACO       # Classe principal
│   │   ├── Edge               # Aresta do grafo com peso e feromônio
│   │   ├── Ant                # Formiga com seleção probabilística
│   │   ├── _acs()             # Variante ACS
│   │   ├── _elitist()         # Variante Elitista
│   │   ├── _max_min()         # Variante MaxMin
│   │   ├── run()              # Executa o modo selecionado
│   │   └── plot()             # Visualiza o melhor tour com matplotlib
│   └── __main__               # Benchmark comparativo → out.csv
│
├── show.py             # Carrega dataset, executa ACO, visualiza com turtle
│
└── tsp dataset/        # Dataset de 225 cidades indianas
```

---

## 10. Conclusão

O projeto demonstra com sucesso a aplicação de três variantes de ACO ao TSP em um dataset real. A implementação é modular — o módulo `aco_tsp.py` aceita qualquer conjunto de pontos 2D e é independente do dataset específico.

As principais decisões de projeto que merecem destaque:

1. **Heurística normalizada** em `_select_node` — garante estabilidade numérica.
2. **Simetria de arestas** compartilhando objetos `Edge` — reduz memória e garante consistência.
3. **Dois regimes em MaxMin** — equilibra exploração inicial com refinamento tardio.
4. **Escala normalizável** em `show.py` — mapeia coordenadas geográficas reais para o espaço de visualização sem perda de fidelidade espacial.

Para trabalhos futuros, sugere-se:

- Substituir a busca linear `in list` por `set` em `_select_node` para ganho de eficiência.
- Avaliar o uso das matrizes de distância por estrada (`road_distance.txt`) como pesos das arestas, aproximando o modelo da realidade logística.
- Implementar critério de parada antecipada por estagnação (sem melhoria por `k` iterações consecutivas).
