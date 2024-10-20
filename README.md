# Pong

Requerimento de torch (pip install torch)

O model1.pth e model2.pth devem ser colocados no mesmo diretório que o bot.py

O bot de pong é dividido em três partes que são ativadas em diferentes situação, uma é usada quando a bola está indo para longe do paddle, nesse caso ele só se posiciona entre a bola que está se afastando dele e o centro, quando a bola está indo em direção ao paddle ele usa as observações para calcular a posição em y que a bola estará quando chegar nele, se ele estiver longe ele vai se mover na direção dela, quando ele estiver perto o suficiente passará para a terceira ia que foi criada usando uma estrutura de
PPO(Proximal policy optimization)


1. Estrutura do PPO:
O código utiliza a técnica PPO para treinar a IA em um ambiente Pong. Existem duas redes neurais, uma para cada jogador, ambas da classe ActorCritic. A arquitetura dessas redes consiste em camadas densamente conectadas que compartilham uma estrutura, mas possuem duas saídas: uma para as ações (cabeçalho de política) e outra para os valores dos estados (cabeçalho de valor).
2. Componentes chave:
ActorCritic: Uma rede neural com duas camadas densas e duas saídas separadas: uma que prevê a política (distribuição de probabilidades das ações) e outra que prevê o valor do estado (estimativa do retorno futuro).
PPOAgent: Classe principal do agente, que contém as funções de seleção de ação, cálculo de vantagem, atualização de política e treinamento.
Funções de vantagem e retorno: A função compute_advantages usa a técnica de GAE (Generalized Advantage Estimation) para calcular vantagens e retornos para cada episódio.
3. Seleção de Ações:
O agente escolhe ações baseadas na política gerada pela rede neural apenas quando está próximo da bola, em outros momentos ele funciona da mesma forma que o bot usado no bot.py. Duas redes distintas (model e model2) são usadas para prever as ações dos dois jogadores (um para cada jogador).
4. Recompensas:
A recompensa foi calculada como reward+=abs(yball-yplayer)*2, yball é a posição da bola quando chega no lado inimigo e yplayer é a posição do paddle inimigo quando a bola foi refletida, ou seja, o agente é recompensado por refletir a bola para a maior
distância posível da posição do inimigo.
5. Atualização do Modelo:
O processo de atualização acontece em blocos de dados coletados durante as interações do agente com o ambiente.
Para cada atualização:
A política atual do agente é comparada com a política anterior usando a razão de probabilidade, ajustada para garantir que as mudanças na política sejam limitadas a um intervalo aceitável (clipping).
A função de perda total combina a perda da política, a perda de valor e uma penalidade de entropia para incentivar a exploração.
6. Cálculo de Retornos e Vantagens:
As vantagens são calculadas com base nas recompensas observadas e nas estimativas de valor fornecidas pela rede neural.
A função compute_advantages calcula tanto as vantagens quanto os retornos esperados, que são usados para ajustar os parâmetros da política (através do algoritmo PPO).
7. Treinamento:
O treinamento ocorre em episódios, onde, em cada episódio, o agente interage com o ambiente por um número definido de passos (num_steps).
A cada passo:
As ações são selecionadas com base nas previsões da rede neural e a resposta do ambiente (recompensas, estados, etc.) é armazenada.
Quando o episódio termina ou o número de passos atinge o limite, os valores armazenados são usados para calcular os retornos e vantagens.
O agente atualiza a política e o valor do estado em várias épocas (num_epochs), usando os dados coletados durante a interação com o ambiente.
8. Parâmetros e Estratégias:
Entropia: Um coeficiente de entropia (entropy_coeff) é adicionado à função de perda para incentivar a exploração. Ao adicionar entropia, o modelo evita se concentrar muito em uma única estratégia.
Gradient Clipping: A técnica de clipping de gradiente (torch.nn.utils.clip_grad_norm_) é usada para evitar grandes atualizações de parâmetros que podem desestabilizar o treinamento.
9. Salvamento dos Modelos:
Ao final do treinamento, os pesos de ambos os modelos (um para cada jogador) são salvos em arquivos com nomes que incluem os parâmetros aleatórios do ambiente. Isso permite identificar e comparar diferentes instâncias de treinamentos com parâmetros diferentes.
