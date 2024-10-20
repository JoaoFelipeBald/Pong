import torch
import numpy as np
from envpong import PongLogic
from torch import nn


# Recriando modelo usado no treino
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer1 = nn.Linear(input_dim, 128)
        self.shared_layer2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):

        x = torch.relu(self.shared_layer1(x))
        x = torch.relu(self.shared_layer2(x))
        return torch.softmax(self.policy_head(x), dim=-1), self.value_head(x)



class BotLeft:
    def __init__(self, env):
        # Cria o modelo usado pelo bot da esquerda
        self.model = ActorCritic(16, 3)
        # Carrega o modelo treinado
        self.model.load_state_dict(torch.load("model1.pth"))  
        self.model.eval() 
        self.env = env
        self.old_vx3=0
        self.old_x3=0
        self.old_y3=0  
        self.append=True
        self.obs = [0] * len(env.observation_space.sample())
        
    def act(self): 
        
        actionp1 = PongLogic.PaddleMove.STILL
        state = self.obs
        self.append1=True
         
         
        # Carrega observações e predições        
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = state
        predicted_y = predict_ball_position(x3, y3, vx3, vy3, x2)
        predicted_y2 = predict_ball_position(x3, y3, vx3, vy3, x1)
         
        # Adiciona predição de onde a bola estará para o model analisar    
            
        state_with_prediction = np.append(state, predicted_y)
        state_with_prediction = np.append(state_with_prediction, predicted_y2)
        
        
        # O bot vai se aproximar da bola caso a bola esteja vindo na direção dele e ele esteja distante
        # se esse bloco não for executado o bloco seguinte será.           
        
        if (vx3>0 and self.old_vx3*vx3>0) or (vx3<0 and abs(y1-predicted_y2)>0.0330) or x3>0.88 or x3<0.13:
            self.append1=False        
            if vx3>0:
                if y1>(predicted_y+0.5)/2:
                    actionp1 = PongLogic.PaddleMove.DOWN
                if y1<(predicted_y+0.5)/2:
                    actionp1 = PongLogic.PaddleMove.UP
                                
            if vx3<0 and abs(y1-predicted_y2)>0.0330:
                            
                if y1>predicted_y2:
                    actionp1 = PongLogic.PaddleMove.DOWN
                if y1<predicted_y2:
                    actionp1 = PongLogic.PaddleMove.UP                           
        
        # Caso o bloco anterior não for executado esse daqui será, nele o model vai tentar achar a melhor posição para refletir a bola para
        # a maior distância possível do outro paddle
                                
        if self.append1:    
            with torch.no_grad():
                action, log_prob = select_action(state_with_prediction, model=self.model)
            if action == 0:
                actionp1 = PongLogic.PaddleMove.DOWN
            elif action == 1:
                actionp1 = PongLogic.PaddleMove.STILL
            elif action == 2:
                actionp1 = PongLogic.PaddleMove.UP
        
        self.old_x3=x3
        self.old_y3=y3
        self.old_vx3=vx3    
            
        return actionp1


    def observe(self, obs):
        self.obs = obs

# Bot da direita, mesmo funcionamento do outro.

class BotRight:
    def __init__(self, env):
        self.model = ActorCritic(16, 3)
        self.model.load_state_dict(torch.load("model2.pth"))  
        self.model.eval()
        self.env = env
        self.obs = [0] * len(env.observation_space.sample())
        self.old_x3=0
        self.old_y3=0
        self.old_vx3=0
        self.append2=True
        
    def act(self):
        actionp2 = PongLogic.PaddleMove.STILL
        state = self.obs
        self.append2=True        
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = state
        predicted_y = predict_ball_position(x3, y3, vx3, vy3, x2)
        predicted_y2 = predict_ball_position(x3, y3, vx3, vy3, x1)
                
        state_with_prediction = np.append(state, predicted_y)
        state_with_prediction = np.append(state_with_prediction, predicted_y2)
        self.old_vx3
                 
        if (vx3<0 and self.old_vx3*vx3>0) or (vx3>0 and abs(y2-predicted_y)>0.0330) or x3>0.88 or x3<0.13:     
            self.append2=False         
            if vx3<0:
                if y2>(predicted_y2+0.5)/2:
                    actionp2 = PongLogic.PaddleMove.DOWN
                if y2<(predicted_y2+0.5)/2:
                    actionp2 = PongLogic.PaddleMove.UP
                                
            if vx3>0 and abs(y2-predicted_y)>0.0330:
                if y2>predicted_y:
                    actionp2 = PongLogic.PaddleMove.DOWN
                if y2<predicted_y:
                    actionp2 = PongLogic.PaddleMove.UP                    
                                
        if self.append2:        
            with torch.no_grad():    
                action2, log_prob = select_action(state_with_prediction, model=self.model)                 
            if action2 == 0:
                actionp2 = PongLogic.PaddleMove.DOWN
            elif action2 == 1:
                actionp2 = PongLogic.PaddleMove.STILL
            elif action2 == 2:
                actionp2 = PongLogic.PaddleMove.UP
        
        self.old_x3=x3
        self.old_y3=y3
        self.old_vx3=vx3    
            
        return actionp2

    def observe(self, obs):
        self.obs = obs


# Função usada para prever posição da bola quando chegar no ponto de determinado paddle
def predict_ball_position(ball_x, ball_y, vel_x, vel_y, paddle_x,  window_height=1):
    # Calcula o tempo para a bola chegar no paddle
    if vel_x==0:
        vel_x=0.00001
    time_to_paddle = (paddle_x - ball_x) / vel_x
    
    # Prediz onde ela estará vericalmente
    predicted_y = ball_y + time_to_paddle * vel_y
    
    # Cuida das vezes que a bola bater na parede
    if predicted_y < 0 or predicted_y > window_height:
        num_bounces = int(predicted_y // window_height)
        if num_bounces % 2 == 0:
            final_y = predicted_y % window_height
        else:
            final_y = window_height - (predicted_y % window_height)
    else:
        final_y = predicted_y

    return final_y

def select_action(state, model):
    state = torch.FloatTensor(state)
    probs, _ = model(state)
    action = np.random.choice(3, p=probs.detach().numpy())
    return action, probs[action]
