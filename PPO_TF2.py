import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import cv2
import pandas as pd

import threading
from threading import Thread, Lock
import time
import sys

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.switch_backend('agg') # needed due to runtime error arising otherwise. See https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


def NN_Model(input_shape, action_space, lr, shared_nn):
    '''This function defines our NN model and its layers and compiles one neural network for each Actor and Critic. It also defines the special PPO loss function'''
    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)

    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)

    if shared_nn:
        value = Dense(1, kernel_initializer='he_uniform')(X)
    else:
        Z = Flatten(input_shape=input_shape)(X_input)
        Z = Dense(512, activation="elu", kernel_initializer='he_uniform')(Z)
        value = Dense(1, kernel_initializer='he_uniform')(Z)

    def ppo_loss(y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:]
        LOSS_CLIPPING = lossclipping
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss =  -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

        return loss

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss=ppo_loss, optimizer=RMSprop(learning_rate=lr))
    print(Actor.summary())

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
    print(Critic.summary())

    return Actor, Critic


class PPO_Agent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        #self.env = gym.make(env_name, render_mode='human') # uncomment this to get a visual representation
        #Note: rendering does not work while thread training
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = eps, 0, -21.0 # pong specific reward
        self.lock = Lock() # lock all to update parameters without other thread interruption
        self.lr = learningrate 
        self.shared_nn = shared_nn

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.EPOCHS = 10 #Number of times the same observations are used to train the NNs 

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = 'APPO_{}'.format(savestring)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = NN_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, shared_nn=self.shared_nn)

    def sample_action(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state, verbose=0)[0] 
        # sample a random choice of the allowed actions for the environment using prediction as probability distribution. 
        action = np.random.choice(self.action_size, p=prediction) 
        return action, prediction

    def discount_rewards(self, reward): #todo: try using a sigmoidal discount
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward) # zero matrix just like reward
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum to ensure that we discount every round, and not every episode (having at least 21 rounds)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        # Standardization helps overal as it shifts the reward curve a bit higher in general. Thus, also cases where the agent survived longer
        # are generally shifted more into the positive, helping the model to learn that surviving is also reward-worthy. 
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards, predictions):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        # Compute advantages
        advantages = discounted_r - values

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards), 
            use_multiprocessing=True)
        self.Critic.fit(states, discounted_r, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards),
            use_multiprocessing=True)
    
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save('Models/' + savestring + '_Actor.h5')
        self.Critic.save('Models/' + savestring + '_Critic.h5')

    # Save model data every 10 episodes
    def RecordData(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-1:] == "0":
            zipped = list(zip(self.episodes, self.scores, self.average))
            reward_data = pd.DataFrame(zipped, columns=['Episode', 'Score', 'Moving Average Score (n=50)'])
            try:
                reward_data.to_csv('Results/mean_rewards_{}.csv'.format(savestring))
            except OSError:
                pass

        return self.average[-1]

    def GetImage(self, frame, image_memory):
        if image_memory.shape == (1,*self.state_size):
            image_memory = np.squeeze(image_memory)

        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        image_memory = np.roll(image_memory, 1, axis = 0)

        # inserting new frame to free space
        image_memory[0,:,:] = new_frame
        
        return np.expand_dims(image_memory, axis=0)

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame, image_memory)
        return state

    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        next_state = self.GetImage(next_state, image_memory)
        return next_state, reward, done, info
    
    def train(self):
        for e in range(self.EPISODES):
            state = self.reset(self.env)
            done, score, SAVING = False, 0, ''
            # Instantiate or reset games memory
            states, actions, rewards, predictions = [], [], [], []
            while not done:
                # Actor picks an action
                action, prediction = self.sample_action(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action, self.env, state)
                # Memorize (state, action, reward) for training
                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                predictions.append(prediction)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.RecordData(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))

                    self.replay(states, actions, rewards, predictions)
                    
        self.env.close()

    def multi_train(self, n_threads):
        '''Function to call train_threading and set up the environment for it'''
        self.env.close()
        # Instantiate one environment per thread
        #envs = [gym.make(self.env_name, render_mode='human') for i in range(n_threads)]
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self,
                    envs[i],
                    i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(3)
            t.start()
            print(f'Threat {t} started.')

        for t in threads:
            time.sleep(10)
            t.join()
            
    def train_threading(self, agent, env, thread):
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.reset(env)
            # Instantiate or reset games memory
            states, actions, rewards, predictions = [], [], [], []
            while not done:
                action, prediction = agent.sample_action(state)
                next_state, reward, done, _ = self.step(action, env, state)

                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                predictions.append(prediction)
                
                score += reward
                state = next_state

            self.lock.acquire()
            self.replay(states, actions, rewards, predictions)
            self.lock.release()

            # Update episode count
            with self.lock:
                average = self.RecordData(score, self.episode)
                # saving best models
                if average >= self.max_average:
                    self.max_average = average
                    self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, SAVING))
                if(self.episode < self.EPISODES):
                    self.episode += 1
        env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(eps):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()


if __name__ == "__main__":

    timer_start = time.time()

    no_threads = 5
    eps = 751
    learningrate = 0.0001 # use lower learning rate than 0.0001 for CNNs
    lossclipping = 0.2
    shared_nn = True # Whether or not the Model is using different neural networks for actor and critic.

    customname = "Dense_512_OutcomeReliabilityComparison_sharedNN"
    t_start = str(datetime.today().strftime('%m%d_%H-%M-%S'))
    savestring = '_'.join([t_start, str(no_threads), str(learningrate), str(lossclipping), customname])

    try: 
        env_name = 'PongDeterministic-v4'
        agent = PPO_Agent(env_name)

        t_start = str(datetime.today().strftime('%m%d_%H-%M-%S'))

        agent.multi_train(n_threads=no_threads) # use as APPO
        #agent.train() # use as PPO
        #agent.test('Models/0823_07-08-10_5_0.0001_0.2_Dense_512_OutcomeReliabilityComparison_2EpochsForNN0_Actor.h5', '')

    except KeyboardInterrupt: # Make the program react to ctrl+c to stop 
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    timer_end = time.time()
    runtime = timedelta(seconds=(timer_end-timer_start))

    print(f"This execution took {runtime} hours.")