import numpy as np #biblioteca que facilita o armazenamento e o acesso a matrizes e vetores de forma otimizada além de proporcinar métodos que facilitam a manipulação dos dados
import math # fornece funcões matemáticas
import matplotlib.pyplot as plt #facilita e prove a geração de gráficos em python
import random # fornece funções para gerar números aleatórios
from genetic_algorithm.routing_utils import haversine, nearest_neighbor, route_distance

LAT = -16.6869
LON = -49.2648

# Variação em torno de Goiânia (em graus)
LAT_RANGE = 0.06
LON_RANGE = 0.06
ORIGIN = (-16.6864, -49.4000)
N_POINTS = 50

pos_range_x=(LAT - LAT_RANGE, LAT + LAT_RANGE)
pos_range_y=(LON - LON_RANGE, LON + LON_RANGE)
points = [(random.uniform(*pos_range_x), random.uniform(*pos_range_y)) for _ in range(N_POINTS)]


cidades = np.array([i for i in range(1, N_POINTS + 1)])
coordenadas_x = np.array([point[0] for point in points])
coordenadas_y = np.array([point[1] for point in points])

numero_cidades = len(cidades)

# fig, ax = plt.subplots()

# for i, txt in enumerate(cidades):
#     ax.annotate(int(txt), (coordenadas_x[i], coordenadas_y[i]))

#     plt.xlabel('Coordenadas X')
# plt.ylabel('Coordenadas Y')
# plt.title('Coordenadas das cidades utilizadas no CV')

# plt.plot(coordenadas_x,coordenadas_y,'ro')
# plt.show()

def distancia_haversine(x,y):
    distancias = np.zeros([numero_cidades,numero_cidades])
    for i in range(len(x)):
        for j in range(len(y)):
            distancias[i][j] = haversine((x[i], y[i]), (x[j], y[j]))
    return distancias


distancias = distancia_haversine(coordenadas_x,coordenadas_y)
print(distancias)

### SETAR PARÂMETROS INICIO ###
#Configuração/Parametrização do algoritmo ABC
tamanhoDaColonia = 10 # tamanho total da colônia de abelhas (empregadas e espectadoras)
metadeDaColonia = int(tamanhoDaColonia/2) #referente ao tamanho da metade da colônia de abelhas
numeroDeTentativas = 10 # número de tantivas que relacionadas a uma solução poder ser melhorada
D = numero_cidades #dimensionalidade do problema em questão ou seja, quantidade de váriaveis de decisão
numeroDeCiclosDeForrageamento = 50 #número de iterações realizados pelas abelhas na colmeia ou ciclo de trabalho delas na busca por mel
numeroDeExecucoes = 10 #número de execuções realizadas pelo algoritmo ABC

colonia = np.zeros([metadeDaColonia, D], dtype=int) #criação da colonia de abelhas onde cada linha representa uma abelha e as colunas os locus de mel referentes as variavéis de decisão do problema abordado pelo ABC
tentativas = np.zeros([metadeDaColonia]) #array que armazenara as tentativas das fontes de alimento evoluirem referente a cada abelha da colmeia
fitnessDaColonia = np.zeros([metadeDaColonia]) #array que armazenara a qualidade das fontes de alimento das abelhas da colmeia


melhorSolucao = np.zeros([D], dtype=int) #melhor solução atual
melhorFitness = 0 #valor de fitness da melhor solução atual

melhoresSolucoes = np.zeros([numeroDeExecucoes,D], dtype=int) #matriz com as melhores soluções encontradas em cada execução do ABC
melhoresFitness  = np.zeros([numeroDeExecucoes]) #array referente as melhores soluções
### SETAR PARÂMETROS FIM ###
def fitness(solucao,distancias):
    valorDoFitness = 0
    for i in range(len(solucao)-1):
        valorDoFitness += distancias[solucao[i],solucao[i+1]]
    valorDoFitness += distancias[solucao[-1],solucao[0]] #retornando da ultima cidade para a cidade inicial
    return valorDoFitness

for i in range(metadeDaColonia):
    colonia[i,:] = np.random.permutation(D) #para cada abelha gerar um roteiro de cidades/variaveis de decisão
    fitnessDaColonia[i] = fitness(colonia[i],distancias)


def selecaoPorRoleta(fitness): #Roleta probabilista refente as aptidoes/fitness da colonia de abelhas
    probs = fitness[:]/fitness.sum()
    escolhida = np.random.uniform(0,probs.sum(),1)
    for i in range(len(probs)):
        if probs[i] > escolhida:
            return i
    return -1


def trocaSwap(novaSolucao):
    posicaoDeTrocas = np.random.randint(D, size=(2)) #gerando 2 posições aleatórias
    while posicaoDeTrocas[0] == posicaoDeTrocas[1]: #garantindo que os números gerados são diferentes
        posicaoDeTrocas = np.random.randint(D, size=(2))
    aux = novaSolucao[posicaoDeTrocas[0]]
    novaSolucao[posicaoDeTrocas[0]] = novaSolucao[posicaoDeTrocas[1]]
    novaSolucao[posicaoDeTrocas[1]] = aux
    return novaSolucao


#definir quem é a abelha com melhor fonte de alimento
melhorSolucao[:] = colonia[fitnessDaColonia.argmin()]
melhorFitness = fitnessDaColonia[fitnessDaColonia.argmin()]

for r in range(numeroDeExecucoes):
    #print("Execucao numero: %d" % (r))
    for iteracao in range(numeroDeCiclosDeForrageamento): #criterio de parada é quantidade de ciclos de forrageamento
        #print("Iteracao: %d" % (iteracao))
        #iniciar fase das abelhas empregadas
        for i in range(metadeDaColonia):
            #gerar uma perturbação na solucao da abelha i e verificar se essa nova solução é melhor que a velha
            novaSolucao = np.zeros([D], dtype=int)
            #aplicando o método simples de troca swap
            novaSolucao[:] = trocaSwap(colonia[i])
            #verificar se a nova solucao tem melhor aptidão que a velha solução/fonte de alimentos da abelha i
            if fitness(novaSolucao,distancias) < fitnessDaColonia[i]:
                colonia[i,:] = novaSolucao
                fitnessDaColonia[i] = fitness(novaSolucao,distancias)
                tentativas[i] = 0 # se a solução foi melhorada reseta o numero de tentativas dessa fonte de alimento
            else:
                tentativas[i] += 1 # se a solução da abelha i não melhorou o array de tentativas na posição i deve ser incrementado representando que a solução i não pode ser melhorada

        #iniciar fase das abelhas espectadoras
        for j in range(metadeDaColonia):   
            abelhaEmpregadaEscolhida = selecaoPorRoleta(fitnessDaColonia)
            while abelhaEmpregadaEscolhida == -1:
                abelhaEmpregadaEscolhida = selecaoPorRoleta(fitnessDaColonia)
            #gerar uma perturbação na solucao da abelha i e verificar se essa nova solução é melhor que a velha
            novaSolucao = np.zeros([D], dtype=int)
            #aplicando o método simples de troca swap na abelha empregada escolhida pelo método de seleção adotado
            novaSolucao[:] = trocaSwap(colonia[abelhaEmpregadaEscolhida])
            #verificar se a nova solucao tem melhor aptidão que a velha solução/fonte de alimentos da abelha i
            if fitness(novaSolucao,distancias) < fitnessDaColonia[i]:
                colonia[i,:] = novaSolucao
                fitnessDaColonia[i] = fitness(novaSolucao,distancias)
                tentativas[i] = 0 # se a solução foi melhorada reseta o numero de tentativas dessa fonte de alimento
            else:
                tentativas[i] += 1 # se a solução da abelha i não melhorou o array de tentativas na posição i deve ser incrementado representando que a solução i não pode ser melhorada

        #iniciar fase da abelha exploradora - só existe uma abelha exploradora na colmeia então ela so realiza uma ação no final de cada ciclo de forrageamento
        #primeiro deve se salvar a melhor abelha do ciclo de forrageamento
        if fitnessDaColonia[fitnessDaColonia.argmin()] < melhorFitness:
            melhorSolucao[:] = colonia[fitnessDaColonia.argmin()]
            melhorFitness = fitnessDaColonia[fitnessDaColonia.argmin()]
        #agora que a melhor solução da iteração foi salva verificamos se a solução de alguma abelha ultrapassou o numero limite de tentativas que elas tinham pra tentar evoluir
        if tentativas[tentativas.argmax()] >= numeroDeTentativas:
            #a função da abelha exploradora é encontrar um nova fonte de alimento para a abelha que ultrapassou seu numero de tentativas na colonia
            colonia[tentativas.argmax()] = np.random.permutation(D) #gera a nova solução
            fitnessDaColonia[tentativas.argmax()] = fitness(colonia[tentativas.argmax()],distancias) #gera o fitness dessa nova solução
            tentativas[tentativas.argmax()] = 0 #reseta o número de tentativas para a nova fonte de alimentos dessa abelha

        #print("Melhor Solucao e Melhor Fitness Atual")
        #print(melhorSolucao)
        #print(melhorFitness)

    melhoresSolucoes[r,:] = melhorSolucao
    melhoresFitness[r] = melhorFitness

print("Fim da Execução!")
for i in range(numeroDeExecucoes):
    print("execucao %d" % (i))
    print(melhoresSolucoes[i,:])
    print(melhoresFitness[i])
    print("-----------------------------")



fig2, ax2 = plt.subplots()

for i, txt in enumerate(cidades):
    ax2.annotate(int(txt), (coordenadas_x[i], coordenadas_y[i]))

    plt.xlabel('Coordenadas X')
plt.ylabel('Coordenadas Y')
plt.title('Instância com 5 cidades')

melhorSolucao = np.append(melhorSolucao[:],melhorSolucao[0])    
plt.plot(coordenadas_x[melhorSolucao[:]],coordenadas_y[melhorSolucao[:]],coordenadas_x[melhorSolucao[:+1]],coordenadas_y[melhorSolucao[:+1]],'ro')
plt.show()

print("Roteiro de cidades percorridas pelo CV = %s" % str(melhorSolucao+1))
print("Distancia percorrida pelo CV = %f" % melhorFitness)

nn_route = nearest_neighbor(ORIGIN, points)
route_distance_nn = route_distance(nn_route, points, ORIGIN)
print("Roteiro de cidades percorridas pelo NN = %s" % str(nn_route))
print("Distancia percorrida pelo NN = %f" % route_distance_nn)