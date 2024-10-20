from envpong import PongLogic
import random

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

# Random bot
class BotRight2:
    def __init__(self, env):
        self.env = env
        self.left=False
        # This bot requires an initial observation, set everything to zero
        self.obs = [0]*len(env.observation_space.sample())
        self.x=0
        self.old_x3=1
        self.old_y3=1
        self.old_vx3=0
        self.coming=False
        
    def act(self):
        
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = self.obs      
        if self.left==False:
            me_x, me_y, mev_x, mev_y, en_x, en_y, env_x, env_y = x2, y2, vx2, vy2m, x1, y1, vx1, vy1
        if self.left==True:
            me_x, me_y, mev_x, mev_y, en_x, en_y, env_x, env_y = x1, y1, vx1, vy1, x2, y2, vx2, vy2m
        
        if (x3-self.old_x3)**2>0.1 or (y3-self.old_y3)**2>0.1 or vx3*self.old_vx3<0:
            if self.left:
                if vx3<0:
                    self.coming=True
                if vx3>0:
                    self.coming=False
            if not self.left:
                if vx3>0:
                    self.coming=True
                if vx3<0:
                    self.coming=False
        
        self.old_x3=x3
        self.old_y3=y3
        self.old_vx3=vx3
        action = PongLogic.PaddleMove.STILL
        if not self.coming:
            if me_y>0.5:
                action = PongLogic.PaddleMove.DOWN
            if me_y<0.5:
                action = PongLogic.PaddleMove.UP
        if self.coming:
            y=round(predict_ball_position(x3,y3,vx3,vy3,me_x,1),4)
            if me_y>y and me_y-y>0.02:
                action = PongLogic.PaddleMove.DOWN
            if me_y<y and y-me_y>0.02:
                action = PongLogic.PaddleMove.UP
        
        return action
    
    def observe(self, obs):
        self.obs = obs
        
# Ball tracking bot
class BotLeft2:
    def __init__(self, env):
        self.env = env
        self.left=True
        # This bot requires an initial observation, set everything to zero
        self.obs = [0]*len(env.observation_space.sample())
        self.x=0
        self.old_x3=1
        self.old_y3=1
        self.old_vx3=0
        self.coming=False
        
    def act(self):
        
        x1, y1, vx1, vy1, x2, y2, vx2, vy2m, x3, y3, vx3, vy3, p1, p2 = self.obs      
        if self.left==False:
            me_x, me_y, mev_x, mev_y, en_x, en_y, env_x, env_y = x2, y2, vx2, vy2m, x1, y1, vx1, vy1
        if self.left==True:
            me_x, me_y, mev_x, mev_y, en_x, en_y, env_x, env_y = x1, y1, vx1, vy1, x2, y2, vx2, vy2m
        
        if (x3-self.old_x3)**2>0.1 or (y3-self.old_y3)**2>0.1 or vx3*self.old_vx3<0:
            if self.left:
                if vx3<0:
                    self.coming=True
                if vx3>0:
                    self.coming=False
            if not self.left:
                if vx3>0:
                    self.coming=True
                if vx3<0:
                    self.coming=False
        
        self.old_x3=x3
        self.old_y3=y3
        self.old_vx3=vx3
        action = PongLogic.PaddleMove.STILL
        if not self.coming:
            if me_y>0.5:
                action = PongLogic.PaddleMove.DOWN
            if me_y<0.5:
                action = PongLogic.PaddleMove.UP
        if self.coming:
            y=round(predict_ball_position(x3,y3,vx3,vy3,me_x,1),4)
            if me_y>y and me_y-y>0.02:
                action = PongLogic.PaddleMove.DOWN
            if me_y<y and y-me_y>0.02:
                action = PongLogic.PaddleMove.UP
        
        return action
    
    def observe(self, obs):
        self.obs = obs
        