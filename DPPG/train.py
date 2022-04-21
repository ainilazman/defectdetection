from numpy import convolve, ones, mean, random
import math
from DDPG import DDPGAgent
from panda_env import PandaEnv, plot_frame_pb
from panda import Panda
import pybullet as p

stepsperepisode = 300 
episodelimit = 5000
period = 200

# Connect to PyBullet in GUI
p.connect(p.GUI, options="--width=2600 --height=1800")

# Initialize Panda environment
env = PandaEnv(
    mu=0.4,  # tangential friction coefficient
    sigma=0.03,  # torsional friction coefficient
    long_finger=False,
    wide_finger=False)

observationSpace = 14  # 7 links 
actionSpace = 2187

agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.shape[0], load_path=load_path) 

episodeCount = 0 
solved = False 

def getPoint():
        x = math.random(-10,10)
        y = math.random(-10,10)
        z = math.random(-10,10)
        return x, y, z


targetPosition = []
for i in 1000:
    targetPosition.append(getPoint())


# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episodeCount < episodelimit:
    observation = env.reset_env()  # Reset robot and get starting observation
    env.episodeScore = 0

    print("===episodeCount:", episodeCount,"===")
    env.target = random.choice(targetPosition)
    # Inner loop is the episode loop
    for step in range(stepsperepisode):
        # In training mode the agent returns the action plus OU noise for exploration
        act = agent.choose_action(observation)

        # get the current selectedAction reward, the new state and whether we reached the
        # the done condition
        newState, reward, done, info = env.step(act*0.032)
        # process of negotiation
        while(newState==["Inprogress"]):
            newState, reward, done, info = env.step([-1])
        
        # Save the current state transition in agent's memory
        agent.remember(state, act, reward, newState, int(done))

        env.episodeScore += reward  # Accumulate episode reward
        # Perform a learning step
        if done or step==stepsperepisode-1:
            # Save the episode's score
            env.episodeScoreList.append(env.episodeScore)
            agent.learn()
            if episodeCount%period==0:
                agent.save_models()
            solved = env.solved()  
            break

        state = newState 

    print("Epoch #", episodeCount, "score:", env.episodeScore)
    fp = open("./exports/Epoch-score.txt","a")
    fp.write(str(env.episodeScore)+'\n')
    fp.close()
    episodeCount += 1  # Increment episode counter

agent.save_models()
if not solved:
    print("Unsolved.")
else:
    print("Solved")

observation= env.reset_env()
env.episodeScore = 0
step = 0
env.target = random.choice(targetPosition)
while True:
    act = agent.choose_action_test(state)
    state, reward, done, _ = env.step(act*0.032)
    # process of negotiation
    while(state==["Inprogress"]):
        state, reward, done, info = env.step([-1])
    
    env.episodeScore += reward  # Accumulate episode reward
    step = step + 1
    if done or step==stepsperepisode-1:
        print("Reward accumulated =", env.episodeScore)
        env.episodeScore = 0
        state = env.reset()
        step = 0
        env.target = random.choice(targetPosition)
