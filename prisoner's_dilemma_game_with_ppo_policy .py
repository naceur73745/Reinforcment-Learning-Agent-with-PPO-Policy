import random

from Env import Prisoners
from AgentPpo import Agent 

"""#Train the Agent"""
import torch
import matplotlib.pyplot as plt
#define a

#2,128,512,2,0.0001,128,2048,0.99,0.4,0.2,0.95
def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda):
  return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda)



def train_function(episode_len  , n_games ,input_dim ,dim1,dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda ):
  env = Prisoners(episode_len,n_games)
  agent  = Create_agent(input_dim ,dim1,dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda )
  for Round in range  (n_games ):
    '''
    if n_games ==  1600 :
      print("we are now in evaluate mode for exploitation ")
      agent.actor.evaluate= True
    '''

    state = env.reset()
    step = 0
    while env.done == False   :


      action , prob , value , distribuation  = agent.choose_action(state)

      new_state  , reward , done , info =  env.step(action)

      print(f"the curent distribuation :{distribuation} , current_state : {state}")

      agent.mem.store_action(state , new_state,action ,reward ,done ,prob ,value )

      agent.learn()

      state = new_state
      print( f" Round  : {Round} , step:{step} ")
      step+= 1



  return env.state_total , env.reward_total , env.chooosed_startegy_each_round ,agent

states  , rewards  , startegies,agent   = train_function(100 ,50  , 2,128,512,2,0.08,128,2048,0.99,0.1,0.2,0.95)

print( "train mode ")

print(states )

print(rewards )

print(startegies )

print("\n")



print("test the agent  ")


def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]
  agent.actor.evaluate==True

  #agent.actor.evaluate =True

  for Round in range  (n_games ):

    state = env.reset()
    step = 0
    while env.done == False   :


      action , prob , value , distribuation  = agent.choose_action(state)
      action , _ , _   = agent.actor.forward(torch.tensor(state ,dtype=torch.float))


      new_state  , reward , done , info =  env.step(action.item())

      print(f"the curent distribuation :{distribuation} , current_state : {state}")

      agent.mem.store_action(state , new_state,action ,reward ,done ,prob ,value )

      agent.learn()

      state = new_state
      print( f" Round  : {Round} , Step:{step} ")
      step+= 1



  return env.state_total , env.reward_total , env.chooosed_startegy_each_round ,agent


eval_states , eval_rewards , startegy  , agent  = test_function(1000 , 1 ,1  , agent)
print("test mode  ")
print(eval_states)
print(eval_rewards)

#function for evaluation
import random

#this function plotts the cooperation rate and defection rate for a player
def plot_bar_chart(rate_of_cooperation, rate_of_defection, title_under_first_plot, title_under_second_plot, title, y_label):
    x = [1, 2]  # x-axis values

    # Heights of the bars
    heights = [rate_of_cooperation, rate_of_defection]

    # Labels for the bars
    labels = [title_under_first_plot, title_under_second_plot]

    # Plotting the bar chart
    plt.bar(x, heights)

    # Adding labels and title
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x, labels)

    # Displaying the plot
    plt.show()


#this  calculates the rewards  of player the reward list should be for only one round

def calculate_reward(reward_list , player_index ):

  reward  = 0
  for state in  reward_list :
    reward+= state[player_index ]
  return reward


#this function used to define the cooperation rate and  and defction rate for player in round and it works only on oponent who alw coop and defct

def coop_defect_rate (state_list ,strategy  ) :

  #we cann add the others startegy also  to test
  right_decision =  0
  false_decision = 0
  for state  in state_list   :

   if strategy == 1: #stands for opent is always defecting  the right decison  will be (1,1)
      if state ==(1,1) :
        right_decision = right_decision+1
      else :
        false_decision = false_decision+1
   else :

      if state ==(1,0) : #in case the oponent always coop  the right decison will eb to alwys defect

        right_decision = right_decision+1
      else :
        false_decision = false_decision+1

  return right_decision , false_decision, strategy

#create an  agent
  def Create_agent(input_dim ,dim1 , dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda):
    return Agent(input_dim ,dim1, dim2 , n_actions , lr  ,butch_size , mem_size , gamma , epsilon_dec  , policy_clip , lamda)
#this plott rewqrds over epochen

#plott reword over epochen
def plot(reward_over_epochen , title  , label , x_label  , y_label ):

    epochs = range(1, len(reward_over_epochen) + 1)
    plt.plot(epochs, reward_over_epochen, 'b', label= label  )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_Reward(reward_over_epochen1,reward_over_epochen2  , title  , label1, label2 , x_label  , y_label ):


    epochs = range(1, len(reward_over_epochen1) + 1)
    plt.plot(epochs, reward_over_epochen1, 'b', label= label1 )
    plt.plot(epochs, reward_over_epochen2, 'r', label= label2  )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()




def test_function (epsiode_len , n_games ,values, agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]

  for Round in  range (n_games ):

    state = env.reset()

    for i in range  (epsiode_len ) :

      state  = agent.convert_function(state )

      action = np.argmax(agent.Q_table[state,:])

      new_state  , reward , done , info =  env.evaluate(action ,values  )

      states.append(new_state)

      rewards.append(info )

      state = new_state

  return states  , rewards

import random

print("test the agent  ")


def test_function (epsiode_len , n_games , agent):

  env = Prisoners(epsiode_len ,n_games)
  states= []
  rewards=[]
  player_reward= []
  player_reward2 =[]
  agent.actor.evaluate==True

  #agent.actor.evaluate =True

  for Round in range  (n_games ):

    state = env.reset()
    step = 0
    sum = 0
    sum2= 0
    while env.done == False   :


      some_action , prob , value , distribuation  = agent.choose_action(state)
      action , _ , _   = agent.actor.forward(torch.tensor(state ,dtype=torch.float))
      new_state  , reward , done , info =  env.evaluate(action.item() ,int(random.choice([0,1])))
      sum+= reward
      sum2+=info[1]
      states.append(new_state)
      rewards.append(info )
      state = new_state
      player_reward.append(sum)
      player_reward2.append(sum2)




  return states , rewards , player_reward, player_reward2



eval_states , eval_rewards ,player_reward,player_reward2  = test_function(1000 , 1   , agent)
print(f" \n\n")
print("*************************test******************************\n")
print(f"the  states  : {eval_states}")
print(f"the  rewards : {eval_rewards}")

print(f"\n")

print(f"\n")
print(f"the Graph for Reward over Epochen ")
print(f" eval  reward : {eval_rewards}" )
plot_Reward(player_reward ,player_reward2,  "reward over epochen " , " AiAgent" ,"Oponent" ,"epochen" ,  "reward")

"""##the agent traiend vs an oponent wich takes  diffrent actions rondomly at the end we notice that the agent learned to defect  learned to cooperate
## if we train the agnet moe and adjust the hyperparameter it can learn to alawys defect wich is the optimal solution
"""

