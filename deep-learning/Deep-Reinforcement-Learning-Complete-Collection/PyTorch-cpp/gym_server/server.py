"""
Contains a class that trains an agent.
"""
import logging
from typing import Tuple
import numpy as np
import gym

from gym_server.envs import make_vec_envs
from gym_server.messages import (InfoMessage, MakeMessage, ResetMessage,
                                 StepMessage)
from gym_server.zmq_client import ZmqClient


RUNNING_REWARD_HORIZON = 10


class Server:
    """
    When `Server.serve()` is called, provides a ZMQ based API for training
    RL agents on OpenAI gym environments.
    """

    def __init__(self, zmq_client: ZmqClient):
        self.zmq_client: ZmqClient = zmq_client
        self.env: gym.Env = None
        logging.info("Gym server initialized")

    def serve(self):
        """
        Run the server.
        """
        logging.info("Serving")
        try:
            self.__serve()
        except KeyboardInterrupt:
            pass

    def _serve(self):
        while True:
            request = self.zmq_client.receive()
            method = request['method']
            param = request['param']

            if method == 'info':
                (action_space_type,
                 action_space_shape,
                 observation_space_type,
                 observation_space_shape) = self.__info()
                self.zmq_client.send(InfoMessage(action_space_type,
                                                 action_space_shape,
                                                 observation_space_type,
                                                 observation_space_shape))

            elif method == 'make':
                self.__make(param['env_name'], param['num_envs'],
                            param['gamma'])
                self.zmq_client.send(MakeMessage())

            elif method == 'reset':
                observation = self.__reset()
                self.zmq_client.send(ResetMessage(observation))

            elif method == 'step':
                if 'render' in param:
                    result = self.__step(
                        np.array(param['actions']), param['render'])
                else:
                    result = self.__step(np.array(param['actions']))
                self.zmq_client.send(StepMessage(result[0],
                                                 result[1],
                                                 result[2],
                                                 result[3]['reward']))

    def info(self):
        """
        Return info about the currently loaded environment
        """
        action_space_type = self.env.action_space.__class__.__name__
        if action_space_type == 'Discrete':
            action_space_shape = [self.env.action_space.n]
        else:
            action_space_shape = self.env.action_space.shape
        observation_space_type = self.env.observation_space.__class__.__name__
        observation_space_shape = self.env.observation_space.shape
        return (action_space_type, action_space_shape, observation_space_type,
                observation_space_shape)

    def make(self, env_name, num_envs, gamma):
        """
        Makes a vectorized environment of the type and number specified.
        """
        logging.info("Making %d %ss", num_envs, env_name)
        self.env = make_vec_envs(env_name, 0, num_envs, gamma)

    def reset(self) -> np.ndarray:
        """
        Resets the environments.
        """
        logging.info("Resetting environments")
        return self.env.reset()

    def step(self,
             actions: np.ndarray,
             render: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        """
        Steps the environments.
        """
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            actions = actions.squeeze(-1)
        observation, reward, done, info = self.env.step(actions)
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            reward = np.expand_dims(reward, -1)
            done = np.expand_dims(done, -1)
        if render:
            self.env.render()
        return observation, reward, done, info

    __info = info
    __make = make
    __reset = reset
    __serve = _serve
    __step = step
