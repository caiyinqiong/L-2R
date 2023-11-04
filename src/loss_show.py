import json
import matplotlib.pyplot as plt
import numpy as np
import os


trainer_state = '/data/users/fyx/caiyinqiong/lifelong_learning/lifelong_lotte/zaug_incre_s2-4_top200_sample100/model_lotte_s2/trainer_state.json'
with open(trainer_state) as f:
    data = json.load(f)
    step = [d['step'] for d in data['log_history'][:-1]]
    loss = [d['loss'] for d in data['log_history'][:-1]]

plt.plot(step, loss)  # 方形
plt.xlabel("step")#横坐标名字
plt.ylabel("loss")#纵坐标名字
save_dir = '/'.join(trainer_state.split('/')[:-1])
plt.savefig(os.path.join(save_dir, 'loss.png'), dpi = 300, bbox_inches='tight')
plt.cla()
