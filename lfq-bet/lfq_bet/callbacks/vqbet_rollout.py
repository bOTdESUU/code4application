from lightning.pytorch.callbacks import Callback
import torch
from collections import deque
import imageio, os
import numpy as np

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from lsq_bet.env.kitchen_env import *
import hydra

class EvalOnEnv(Callback):
    def __init__(self, 
                 env, goal_fn):
        super().__init__()
        self.env = env
        self.goal_fn = goal_fn
        self.video_record = VideoRecorder(dir_name='./logs/debug/video')
        self.eval_per_epoch = 50

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.eval_per_epoch == 0:
            self.evaluate(trainer, pl_module)

    def evaluate(self, trainer, pl_module):
        #TODO:get the parameter right using config
        env = self.env
        goal_fn = self.goal_fn
        cbet_model = pl_module
        videorecorder = self.video_record
        action_window_size = 1
        num_evals = 10
        num_eval_per_goal = 1
        eval_window_size = 10
        num_video = 2
        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []  # only used in pusht env
        avg_final_coverage = []  # only used in pusht env
        for goal_idx in range(num_evals):
            if videorecorder is not None:
                videorecorder.init(enabled=(goal_idx <= num_video)) 
            for _ in range(num_eval_per_goal):
                obs_stack = deque(maxlen=eval_window_size)
                obs_stack.append(env.reset())
                done, step, total_reward = False, 0, 0
                goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                while not done:
                    obs = torch.from_numpy(np.stack(obs_stack)).float().to(pl_module.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=pl_module.device)
                    #TODO: modify action here,not act[-1]
                    action, _, _ = cbet_model.eval_on_vqbet_env(obs.unsqueeze(0), goal.unsqueeze(0), None)
                    if action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    obs, reward, done, info = env.step(curr_action)
                    if videorecorder.enabled:
                        videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    obs_stack.append(obs)
                    # if "pusht" not in config_name:
                    if True:
                        goal, _ = goal_fn(env, obs_stack[-1], goal_idx, step)
                avg_reward += total_reward
                # if "pusht" in config_name:
                #     env.env._seed += 1
                #     avg_max_coverage.append(info["max_coverage"])
                #     avg_final_coverage.append(info["final_coverage"])
                # completion_id_list.append(info["all_completions_ids"])
            epoch = trainer.current_epoch
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
            pl_module.logger.experiment.log({"eval/reward" :avg_reward / (num_evals * num_eval_per_goal)})
            
        # return (
        #     avg_reward / (num_evals * num_eval_per_goal),
        #     completion_id_list,
        #     avg_max_coverage,
        #     avg_final_coverage,
        # )


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs):
        if self.enabled:
            self.frames.append(obs)
            # self.frames.append(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

# def test_callback():
#     # Create a dummy environment and goal function
#     env = DummyEnvironment()
#     goal_fn = DummyGoalFunction()

#     # Create an instance of the callback
#     callback = EvalOnEnv(env=env, goal_fn=goal_fn)

#     # Create a dummy trainer and pl_module
#     trainer = DummyTrainer()
#     pl_module = DummyModule()

#     # Call the on_validation_end method of the callback
#     callback.on_validation_end(trainer, pl_module)