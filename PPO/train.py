import math
import numpy as np
import time
import matplotlib.pyplot as plt

from panda_env import PandaEnv, plot_frame_pb
from panda import Panda
from PPO import PPO
import pybullet as p

# Connect to PyBullet in GUI
p.connect(p.GUI, options="--width=2600 --height=1800")
p.resetDebugVisualizerCamera(0.5, 135, -30, [0.5, 0, 0])

# Initialize Panda environment
env = PandaEnv(
    mu=0.4,  # tangential friction coefficient
    sigma=0.03,  # torsional friction coefficient
    long_finger=False,
    wide_finger=False)

observationSpace = 14  # 7 links 
actionSpace = 2187
agent = PPO(observationSpace, actionSpace)

solved = False
#cnt_veryClose = 0

episodeCount = 0  # Episode counter
episodeLimit = 3000  
stepsPerEpisode = 100 
episodeScoreList = []  
doneTotal = 0

ee_pos, ee_orn = env.get_ee()

def getPoint():
        x = math.random(-10,10)
        y = math.random(-10,10)
        z = math.random(-10,10)
        return x, y, z

def solved():
		if len(episodeScoreList) > 1000:  # O
			if np.mean(episodeScoreList[-100:]) > 1000:  # Last 100 episodes' scores average value
				return True
		return False


targetPosition = []
for i in 1000:
    targetPosition.append(getPoint())


preL2norm = np.linalg.norm([targetPosition[0]-ee_pos[0],targetPosition[1]-ee_pos[1],targetPosition[2]-ee_pos[2]])

while not solved and episodeCount < episodeLimit:
	# Reset robot and get starting observation
    observation = env.reset_env()
    episodeScore = 0
	cnt_veryClose = 0
    for step in range(stepsPerEpisode):
		print("step: ", step)
		# Samples from the probability distribution
		selectedAction, actionProb = agent.work(observation, type_="selectAction")
		
		# Step to get current state
		newObservation, reward, done, info = env.step([selectedAction])
		print("L2",newObservation[-1])

		if newObservation[-1] <= 0.01:
			cnt_veryClose += 1
		if cnt_veryClose >= 50 or step==env.stepsPerEpisode-1:
			done = True
			env.preL2norm=0.4
		# compute reward here
	
		if newObservation[0]-(-2.897)<0.05 or 2.897-newObservation[0]<0.05 or\
			newObservation[1]-(-1.763)<0.05 or 1.763-newObservation[1]<0.05 or\
			newObservation[2]-(-2.8973)<0.05 or 2.8973-newObservation[2]<0.05 or\
			newObservation[3]-(-3.072)<0.05 or -0.0697976-newObservation[3]<0.05 or\
			newObservation[4]-(-2.8973)<0.05 or 2.8973-newObservation[4]<0.05 or\
			newObservation[5]-(-0.0175)<0.05 or 3.7525-newObservation[5]<0.05 or\
			newObservation[6]-(-2.897)<0.05 or 2.897-newObservation[6]<0.05:
			reward = -1 # if on of the motors on the limit, reward = -2
		else:
			if(newObservation[-1]<0.01):
				reward = 10 #*((stepsPerEpisode - step)/stepsPerEpisode) 
			elif(newObservation[-1]<0.05):
				reward = 5 #*((stepsPerEpisode - step)/stepsPerEpisode)
			elif(newObservation[-1]<0.1):
				reward = 1 #*((stepsPerEpisode - step)/tepsPerEpisode)
			else:
				reward = -(newObservation[-1]-preL2norm) # negative reward
			preL2norm = newObservation[-1]
 
		print("reward: ",reward)
		print("L2norm: ", newObservation[-1])
		print("tarPosition(trans): ", newObservation[7:10])
		print("endPosition: ", newObservation[10:13])
	
		# ------compute reward end------
		# Save the current state transition in agent's memory
		trans = (observation, selectedAction, actionProb, reward, newObservation)
		agent.storeTransition(trans)

		
		if done:
			if(step==0):
				print("0 Step but done?")
				continue
			print("done gogo")
			# Save the episode's score
			episodeScoreList.append(episodeScore)
			agent.trainStep(batchSize=step)
			solved = solved()  # Check whether the task is solved
			agent.save('')
			break
		
		episodeScore += reward  # Accumulate episode reward
		observation = newObservation  # observation for next step is current step's newObservation
		
	fp = open("Epoch-score.txt","a")
	fp.write(str(Epoch-score)+'\n')
	fp.close()
	print("Episode #", episodeCount, "score:", Epoch-score)
	episodeCount += 1  # Increment episode counter

if not solved:
	print("Not solved")
elif solved:
	print("Solved")
	
observation = env.reset()

while True:
	selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
	observation, _, _, _ = env.step([selectedAction])
