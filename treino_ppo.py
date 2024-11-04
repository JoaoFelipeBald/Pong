# Daniel Cavalcanti Jeronymo
# Pong game  using Arcade library with Neural Network training
# 08/2020
# 
#
import arcade
import os
import time
import threading
import time
import random
import math
import numpy as np
import itertools
from enum import IntEnum
import copy
import arcade

import gym
from gym.spaces import Tuple,Box,Discrete,MultiDiscrete
def predict_ball_position(ball_x, ball_y, vel_x, vel_y, paddle_x,  window_height):
    # Step 1: Calculate time to reach paddle
    if vel_x==0:
        vel_x=0.00001
    time_to_paddle = (paddle_x - ball_x) / vel_x
    
    # Step 2: Predict where ball will be vertically
    predicted_y = ball_y + time_to_paddle * vel_y
    
    # Step 3: Handle wall bounces
    if predicted_y < 0 or predicted_y > window_height:
        # Calculate how many times the ball bounces
        num_bounces = int(predicted_y // window_height)
        
        # Reflect the y-position depending on even or odd number of bounces
        if num_bounces % 2 == 0:
            final_y = predicted_y % window_height
        else:
            final_y = window_height - (predicted_y % window_height)
    else:
        final_y = predicted_y

    return final_y

class Rect:
    def __init__(self, center, shape):
        self.center = center
        self.shape = shape

        self.box = self.calculateBox()
        self.vertices = self.calculateVertices()

        self.box = np.array(self.box)
        self.vertices = np.array(self.vertices)

    def calculateBox(self):
        offsets = []

        for x,width in zip(self.center, self.shape):
            offsets.append( (x - width/2, x + width/2) )

        return offsets

    def calculateVertices(self):
        return [v for v in itertools.product(*self.box)]

    # Simple intersect checking for vertices of other inside this box
    def intersect(self, other):
        # for each vertex of "other", check if it's inside this shape's box
        for v in other.vertices:
            if ((self.box[:,0] < v) & (v < self.box[:,1])).all():
                return True

        return False

class PongLogic:
    class PaddleMove(IntEnum):
        UP = 1
        STILL = 0
        DOWN = -1

    class GameState():
        def __init__(self, player1action, player2action, paddle1Position, paddle2Position, paddle1Velocity, paddle2Velocity, ballPosition, ballVelocity, player1Score, player2Score, time, totalTime):
            self.paddle1Position = paddle1Position
            self.paddle2Position = paddle2Position
            self.paddle1Velocity = paddle1Velocity
            self.paddle2Velocity = paddle2Velocity

            self.ballPosition = ballPosition
            self.ballVelocity = ballVelocity

            self.player1action = player1action
            self.player2action = player2action

            self.player1Score = player1Score
            self.player2Score = player2Score

            self.time = time
            self.totalTime = totalTime

    def randomBallVelocity(self, mag):
        maxAngle = math.radians(60) # 60 degrees

        r = np.random.uniform(-1, 1, 1)[0]*maxAngle
        leftRight = np.random.choice([-1,1])

        r1 = mag*leftRight*math.cos(r)
        r2 = mag*math.sin(r)

        v = np.array([r1, r2], dtype=np.float64)
        return v

    def bounceBallTop(self, state):
        state.ballVelocity[1] = -state.ballVelocity[1]
        #state.ballVelocity *= BOUNCE_FACTOR

        state.ballPosition += state.ballVelocity*self.dt

    def bounceBallPaddle(self, id, state):
        maxAngle = math.pi/3 # 60 degrees
        BOUNCE_FACTOR = 1.1
        ballSpeed = np.linalg.norm(state.ballVelocity)

        if id == 1:
            paddleposY = state.paddle1Position[1]
        else:
            paddleposY = state.paddle2Position[1]

        # get intersect in range [-1,1] (ish... there will be an error from bad rectangle intersect)
        relativeIntersect = (state.ballPosition[1] - paddleposY) / (self.paddleShape[1]/2 + self.ballShape[1]/2)
        bounceAngle = relativeIntersect * maxAngle

        
        # invert bounce angle for player 2

        state.ballVelocity[0] = ballSpeed*BOUNCE_FACTOR*math.cos(bounceAngle)
        state.ballVelocity[1] = ballSpeed*BOUNCE_FACTOR*math.sin(bounceAngle)
        '''
        if state.ballPosition[0]>200:
            print(state.ballVelocity[0])
            print(state.ballVelocity[1])
        '''    
            

        # mirror bounce for player 2
        if id == 2:
            state.ballVelocity[0] *= -1

        # Clamp speed for numerical purposes
        state.ballVelocity = np.minimum( state.ballVelocity , self.ballVelocityMag*100)
        state.ballVelocity = np.maximum( state.ballVelocity , -self.ballVelocityMag*100)

        state.ballPosition += state.ballVelocity*self.dt


    def __init__(self, dt, windowShape, paddleShape, paddleOffset, paddleVelocity, ballShape, ballPosition, ballVelocityMag, debugPrint=True):
        # Game constants
        self.windowWidth, self.windowHeight = windowShape

        # Drawing bounds
        self.boundTop = 0
        self.boundBottom = self.windowHeight
        self.boundLeft = 0
        self.boundRight = self.windowWidth

        self.paddleShape = paddleShape
        self.paddleVelocity = np.array([0, paddleVelocity])

        self.ballShape = ballShape
        self.ballVelocityMag = ballVelocityMag

        # Initial game state
        paddle1Position = np.array([self.windowWidth*paddleOffset, self.windowHeight//2])
        paddle2Position = np.array([self.windowWidth*(1-paddleOffset), self.windowHeight//2])
        paddle1Velocity = np.zeros(2)
        paddle2Velocity = np.zeros(2)
        ballPosition = np.array(ballPosition)
        ballVelocity = self.randomBallVelocity(ballVelocityMag)
        self.s0 = self.GameState(PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.STILL, paddle1Position, paddle2Position, paddle1Velocity, paddle2Velocity, ballPosition, ballVelocity, 0, 0, 0.0, 0.0)

        # Game state history
        self.states = [self.s0]

        # Time step
        self.dt = dt
        
        self.debugPrint = debugPrint

    # Restores initial state
    def reset(self, winnerId):
        # Make a copy of initial state
        state = copy.deepcopy(self.s0)

        # Keep player scores and time
        state.player1Score = self.states[-1].player1Score + (winnerId == 1)
        state.player2Score = self.states[-1].player2Score + (winnerId == 2)
        state.totalTime = self.states[-1].totalTime

        # Modify ball velocity from zero state
        state.ballVelocity = self.randomBallVelocity(self.ballVelocityMag)
        # Append to states list as the current state
        self.states += [state]

        if self.debugPrint:
            print("---SCORE---")
            print("Player 1: ", self.states[-1].player1Score)
            print("Player 2: ", self.states[-1].player2Score)


    # TODO p1 and p2 actions could be replaced by an action list
    def update(self, player1action, player2action):
        # Get current state
        state = copy.deepcopy(self.states[-1])

        # Update clock
        state.time += self.dt
        state.totalTime += self.dt

        # Store actions in state
        state.player1action, state.player2action = player1action, player2action

        # Update paddles
        state.paddle1Velocity = self.paddleVelocity*player1action
        state.paddle2Velocity = self.paddleVelocity*player2action

        state.paddle1Position += state.paddle1Velocity*self.dt
        state.paddle2Position += state.paddle2Velocity*self.dt

        # For collision checks
        paddle1Rect = Rect(state.paddle1Position, self.paddleShape)
        paddle2Rect = Rect(state.paddle2Position, self.paddleShape)
        ballRect = Rect(state.ballPosition, self.ballShape)

        # Limit paddles to stay inside screen
        paddleOffset = paddle1Rect.box[1][0] - self.boundTop
        if paddleOffset < 0:
            state.paddle1Position[1] -= paddleOffset

        paddleOffset = paddle2Rect.box[1][0] - self.boundTop
        if paddleOffset < 0:
            state.paddle2Position[1] -= paddleOffset

        paddleOffset = self.boundBottom - paddle1Rect.box[1][1]
        if paddleOffset < 0:
            state.paddle1Position[1] += paddleOffset

        paddleOffset = self.boundBottom - paddle2Rect.box[1][1]
        if paddleOffset < 0:
            state.paddle2Position[1] += paddleOffset

        # Update ball
        state.ballPosition += state.ballVelocity*self.dt

        # Check bouncing on top or bottom limits
        if state.ballPosition[1] <= self.boundTop or state.ballPosition[1] >= self.boundBottom:
            self.bounceBallTop(state)

        # Check bouncing on paddles
        if paddle1Rect.intersect(ballRect) or ballRect.intersect(paddle1Rect):
            self.bounceBallPaddle(1, state)

        if paddle2Rect.intersect(ballRect) or ballRect.intersect(paddle2Rect):
            self.bounceBallPaddle(2, state)

        # Add state to state history
        self.states += [state]

        # Check win conditions
        if state.ballPosition[0] < self.boundLeft:
            self.reset(2)

        if state.ballPosition[0] > self.boundRight:
            self.reset(1)
            
            
from gym.spaces import Box, Discrete, Tuple

class PongEnv(gym.Env):
    def __init__(self, width=400, height=400, FPS=30.0, debugPrint=False):
        super().__init__()
        self.old_x3 = 0.5
        self.old_y3 = 0.5
        self.old_vx3 = 0
        self.old_points1=0
        self.old_points2=0
        self.createGame(width, height, FPS, debugPrint)
        self.bounces=0
        
        # Define action space
        self.action_space = Discrete(3, start=-1)

        # Define observation space
        self.observation_space = Tuple((
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 1 x position
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 1 y position
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 1 x velocity
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 1 y velocity
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 2 x position
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 2 y position
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 2 x velocity
            Box(0, 1, shape=(1,), dtype=np.float32),  # paddle 2 y velocity
            Box(0, 1, shape=(1,), dtype=np.float32),  # ball x position
            Box(0, 1, shape=(1,), dtype=np.float32),  # ball y position
            Box(0, 1, shape=(1,), dtype=np.float32),  # ball x velocity
            Box(0, 1, shape=(1,), dtype=np.float32),  # ball y velocity
            Discrete(3, start=-1),  # player 1 action
            Discrete(3, start=-1)   # player 2 action
        ))


            
    def step(self, actionp1, actionp2):
        reward = 0.001
        reward2=0.001
        done = False
        truncated = False
        info = {}
        self.game.update(actionp1, actionp2)

              
              
        # update observation
        obs = self.getInputs(self.game.states[-1])
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = obs      
        predicted_y = predict_ball_position(x3, y3, vx3, vy3, x2, 1)
        predicted_y2 = predict_ball_position(x3, y3, vx3, vy3, x1, 1)
        
        if vx3>0:
            reward+=abs(y2-predicted_y)*(1-abs(x3-x2))*2
            reward2-=abs(y2-predicted_y)*(1-abs(x3-x2))*6
        if vx3<0:
            reward-=abs(y1-predicted_y2)*(1-abs(x3-x1))*2
            reward2+=abs(y1-predicted_y2)*(1-abs(x3-x1))*6
        
        #print(self.bounces)    
        
        
        if x3<0.13:
            reward-=abs(y1-predicted_y2)*20
            reward2+=abs(y1-predicted_y2)*20
            
        if x3>0.88:
            reward+=abs(y2-predicted_y)*20
            reward2-=abs(y2-predicted_y)*20
        
        #print("r",reward)
        #print("r2",reward2)
            
        if (x3-self.old_x3)**2<0.2 and (y3-self.old_y3)**2<0.2 and vx3*self.old_vx3<0:
            self.bounces+=1                             
        self.old_x3=x3
        self.old_y3=y3
        self.old_vx3=vx3
  
        self.steps += 1
        self.old_points1=self.game.states[-1].player1Score
        self.old_points2=self.game.states[-1].player2Score
        
        
        if self.game.states[-1].player1Score +self.game.states[-1].player2Score > 6:
            done = True
            print(self.bounces) 
            self.bounces=0
            self.old_points1=0
            self.old_points2=0
        return obs, reward, reward2, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # use same parameters from last game
        self.createGame(self.game.windowWidth, self.game.windowHeight, 1/self.game.dt, self.game.debugPrint)

        obs = self.getInputs(self.game.states[-1])
        info = {}

        return obs, info

    def render(self):
        # Unused
        pass
    
        #if not self.screen:
        #    return
        
    def createGame(self, width, height, simFPS, debugPrint):
        # Setup game        
        self.game = PongLogic(1/simFPS, windowShape=(width, height), paddleShape=(10,30), paddleOffset=0.15, paddleVelocity=200, ballShape=(5,5), ballPosition=(width/2,height/2), ballVelocityMag=100, debugPrint=debugPrint)
        self.steps = 0

    # Serialize a given state into relevant inputs
    def getInputs(self, state):
        inputs = []
        inputs += [state.paddle1Position[0]/self.game.windowWidth]
        inputs += [state.paddle1Position[1]/self.game.windowHeight]
        inputs += [state.paddle1Velocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle1Velocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle2Position[0]/self.game.windowWidth]
        inputs += [state.paddle2Position[1]/self.game.windowHeight]
        inputs += [state.paddle2Velocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.paddle2Velocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.ballPosition[0]/self.game.windowWidth]
        inputs += [state.ballPosition[1]/self.game.windowHeight]
        inputs += [state.ballVelocity[0]/(self.game.ballVelocityMag*100)]
        inputs += [state.ballVelocity[1]/(self.game.ballVelocityMag*100)]
        inputs += [state.player1action]
        inputs += [state.player2action]

        return inputs
    
class PongGUIEnv(arcade.Window, PongEnv):
    def __init__(self, width=400, height=400, FPS=30.0):
        arcade.Window.__init__(self, width, height, 'CYBERPONG')
        PongEnv.__init__(self)
        
        self.set_update_rate(1/FPS)

        arcade.set_background_color(arcade.color.ARSENIC)

        self.player1action = PongLogic.PaddleMove.STILL
        self.player2action = PongLogic.PaddleMove.STILL

    def on_draw(self):
        arcade.start_render()

        paddle1Position = self.game.states[-1].paddle1Position
        paddle2Position = self.game.states[-1].paddle2Position
        ballPosition = self.game.states[-1].ballPosition

        arcade.draw_rectangle_filled(paddle1Position[0], paddle1Position[1], self.game.paddleShape[0], self.game.paddleShape[1], arcade.color.WHITE_SMOKE)
        arcade.draw_rectangle_filled(paddle2Position[0], paddle2Position[1], self.game.paddleShape[0], self.game.paddleShape[1], arcade.color.WHITE_SMOKE)
        arcade.draw_rectangle_filled(ballPosition[0], ballPosition[1], self.game.ballShape[0], self.game.ballShape[1], arcade.color.WHITE_SMOKE)

    def update(self, dt):
        # game update must be called manually by simulation loop in client application from a separate thread from Arcade
        #self.game.update(self.player1action, self.player2action)
        pass 

    def on_key_press(self, key, key_modifiers):
        if key == arcade.key.W:
            self.player1action = PongLogic.PaddleMove.UP
        elif key == arcade.key.S:
            self.player1action = PongLogic.PaddleMove.DOWN

        if key == arcade.key.UP:
            self.player2action = PongLogic.PaddleMove.UP
        elif key == arcade.key.DOWN:
            self.player2action = PongLogic.PaddleMove.DOWN
            


    def on_key_release(self, key, key_modifiers):
        if key == arcade.key.W or key == arcade.key.S:
            self.player1action = PongLogic.PaddleMove.STILL

        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.player2action = PongLogic.PaddleMove.STILL
        
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer1 = nn.Linear(input_dim, 128)
        self.shared_layer2 = nn.Linear(128, 128)  # New hidden layer
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # Multiple layers with ReLU activations
        x = torch.relu(self.shared_layer1(x))
        x = torch.relu(self.shared_layer2(x))  # Add another layer here
        return torch.softmax(self.policy_head(x), dim=-1), self.value_head(x)

class A2CAgent:
    def __init__(self, env):
        torch.autograd.set_detect_anomaly(True)
        self.env = env
        self.action_dim = env.action_space.n
        
        # Calculate the total number of observation dimensions
        self.obs_dim = sum(space.shape[0] if isinstance(space, Box) else 1 for space in env.observation_space)
        # Initialize the Actor-Critic model with the correct dimensions

        self.model = ActorCritic(self.obs_dim+2, self.action_dim)
        self.model2 = ActorCritic(self.obs_dim+2, self.action_dim)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr=0.001)
        
        self.gamma = 0.99  # Discount factor



    def select_action(self, state, model):

        state = torch.FloatTensor(state)
        probs, _ = model(state)
        action = np.random.choice(self.action_dim, p=probs.detach().numpy())
        return action

    def update(self, rewards, log_probs, values, next_value, player, probs_list):
        returns = []
        optimizer = self.optimizer if player == 1 else self.optimizer2
        R = next_value
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        advantage = returns - values.detach()

        # Policy loss
        policy_loss = -log_probs.clone() * advantage.clone()

        # Value loss
        value_loss = advantage.clone().pow(2)

        # Entropy loss (to encourage exploration)
        entropy = -torch.sum(probs_list * torch.log(probs_list + 1e-8), dim=-1).mean()
        entropy_loss = -0.01 * entropy  # The coefficient here can be tuned

        # Combine losses
        total_loss = (policy_loss + value_loss).mean() + entropy_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # For model 1
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 0.5)  # For model 2
        optimizer.step()


def train_agent(agent, num_episodes):
    torch.autograd.set_detect_anomaly(True)
    for episode in range(num_episodes):
        state, _ = agent.env.reset()  # Corrected: Unpack the reset
        done = False
        rewards = []
        rewards2 = []
        log_probs = []
        log_probs2 = []
        values = []
        values2 = []
        probs_list=[]
        probs_list2=[]

        while not done:
            # Use your prediction function
            x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = state
            predicted_y = predict_ball_position(x3, y3, vx3, vy3, x2, 1)
            predicted_y2 = predict_ball_position(x3, y3, vx3, vy3, x1, 1)

            # Append the predicted ball position to the state
            state_with_prediction = np.append(state, predicted_y)
            state_with_prediction = np.append(state_with_prediction, predicted_y2)

            # Convert to a torch tensor for the model
            state_tensor = torch.FloatTensor(state_with_prediction)
            # Inside the training loop
            probs, value = agent.model(state_tensor)
            probs2, value2 = agent.model2(state_tensor)

            # Select actions
            action_p1 = np.random.choice(agent.action_dim, p=probs.detach().numpy())
            action_p2 = np.random.choice(agent.action_dim, p=probs2.detach().numpy())

            # Convert actions to Pong logic
            if action_p1 == 0:
                action_p1 = PongLogic.PaddleMove.DOWN
            elif action_p1 == 1:
                action_p1 = PongLogic.PaddleMove.STILL
            elif action_p1 == 2:
                action_p1 = PongLogic.PaddleMove.UP

            if action_p2 == 0:
                action_p2 = PongLogic.PaddleMove.DOWN
            elif action_p2 == 1:
                action_p2 = PongLogic.PaddleMove.STILL
            elif action_p2 == 2:
                action_p2 = PongLogic.PaddleMove.UP
            log_prob = torch.log(probs[action_p1])
            log_prob2 = torch.log(probs2[action_p2])

            # Take action and get rewards
            next_state, reward, reward2, done, _, _ = agent.env.step(action_p1, action_p2)

            # Store everything for player 1
            rewards.append(reward)
            log_probs.append(log_prob.clone().unsqueeze(0))
            values.append(value.clone())
            probs_list.append(probs.clone())  # Store probs for entropy calculation

            # Store everything for player 2
            rewards2.append(reward2)
            log_probs2.append(log_prob2.clone().unsqueeze(0))
            values2.append(value2.clone())
            probs_list2.append(probs2.clone())  # Store probs for entropy calculation

            state = next_state
            #time.sleep(0.6)

        # Last value
        print("was done")
        state= next_state
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = state
        predicted_y = predict_ball_position(x3, y3, vx3, vy3, x2, 1)
        predicted_y2 = predict_ball_position(x3, y3, vx3, vy3, x1, 1)

        # Append the predicted ball position to the state
        state_with_prediction = np.append(state, predicted_y)
        state_with_prediction = np.append(state_with_prediction, predicted_y2)
  
        # Convert to a torch tensor for the model
        state_tensor = torch.FloatTensor(state_with_prediction)

        _, next_value = agent.model(torch.FloatTensor(state_tensor))
        # Update for player 1
        # Convert list of probs to tensor before passing it to update
        probs_tensor = torch.cat(probs_list)
        agent.update(rewards, log_probs, values, next_value, player=1, probs_list=probs_tensor)

        _, next_value2 = agent.model2(torch.FloatTensor(state_tensor))
        # Update for player 2
        # Convert list of probs to tensor before passing it to update
        probs_tensor2 = torch.cat(probs_list2)
        agent.update(rewards2, log_probs2, values2, next_value2, player=2, probs_list=probs_tensor2)


        if episode % 10 == 0:
            ...


if __name__ == "__main__":

    env = PongGUIEnv()  # Your Pong environment instance
    agent = A2CAgent(env)
    #train_agent(agent, num_episodes=200)
    threading.Thread(target=train_agent, args=(agent,500)).start()
    arcade.run()
    torch.save(agent.model.state_dict(), 'model.pth')
    torch.save(agent.model2.state_dict(), 'model2.pth')
    print(f"Entire model saved")
    
