import numpy as np
from dm_control.utils.rewards import tolerance
import torch
from scipy.spatial.transform import Rotation

def go2_straight_walk_reward(obs, action):
    """
    奖励函数设计目标：
    1. 鼓励机器人向前行走（最大化基座前进速度）。
    2. 保持机器人姿态稳定（最小化基座俯仰角、侧倾角）。
    3. 减少能量消耗（最小化关节扭矩）。
    4. 避免不稳定的运动（惩罚关节速度过大）。
    5. 保持足端接触稳定（惩罚滑移或悬空）。

    参数：
    - obs: 62 维观察向量，由 _get_obs 返回。
    - action: 当前动作向量，形状为 (12,)，表示关节力矩。

    返回：
    - reward: 综合奖励值
    """
    # 分解观察值（假设 obs 结构如下）
    base_linear_velocity = obs[31:34]  # 基座线速度 (x, y, z)
    base_orientation = obs[3:7]        # 基座姿态（四元数）
    q_joints = obs[7:19]               # 关节角度 (12,)
    dq_joints = obs[19:31]             # 关节速度 (12,)
    foot_contact = obs[37:41]          # 足端接触信息 (4,)
    joint_torques = obs[44:62]         # 关节力矩 (18,)

    # 1. 前进速度奖励（基座 x 方向线速度）
    forward_vel = base_linear_velocity[0]  # 假设 x 方向为前进方向
    vel_reward = 10.0 * forward_vel

    # 2. 姿态稳定惩罚（从四元数计算俯仰角、侧倾角）
    # 四元数转换为欧拉角（假设使用 ZYX 顺序）
    
    
    #if np.linalg.norm(base_orientation) == 0:
     #   base_orientation = np.array([1, 0, 0, 0])  # 设置为单位四元数
    #rot = Rotation.from_quat(base_orientation)
    #pitch, roll, _ = rot.as_euler('zyx', degrees=False)  # 俯仰角、侧倾角
    #posture_penalty = 2.0 * (pitch**2 + roll**2)

    # 3. 能量消耗惩罚（关节力矩的平方和）
    energy_cost = 0.01 * np.sum(np.square(joint_torques))

    # 4. 关节速度惩罚（避免不稳定的运动）
    joint_speed_penalty = 0.005 * np.sum(np.square(dq_joints))

    # 5. 足端接触稳定性惩罚（鼓励足端稳定接触地面）
    # 若足端接触标志为 0（悬空），则惩罚
    foot_air_penalty = 0.1 * np.sum(1.0 - foot_contact)

    # 综合奖励
    reward = (
        vel_reward 
        - energy_cost 
        #- posture_penalty
        - joint_speed_penalty 
        - foot_air_penalty
    )
    print(f"reward:",reward)
    return reward
#robot = Go1EnvWithBounds()

def go2_straight_walk_reward_low(obs, action):
    """Go2机器人直行任务的低层次奖励函数
    设计目标：在保持稳定的前提下实现快速直行
    
    观测特征说明（13维）：
    [0]基座高度  [1]基座俯仰角
    [2]X线速度  [3]Y线速度  [4]俯仰角速度
    [5-6]前腿角度差  [7-8]前腿速度均值
    [9-10]后腿角度差  [11-12]后腿速度均值
    
    奖励组成：
    1. 前进速度奖励：鼓励接近目标速度（0.8m/s）
    2. 运动稳定性惩罚：高度、姿态、侧向运动
    3. 关节对称性奖励：促进协调步态
    4. 能量效率惩罚：抑制剧烈动作
    5. 生存奖励：鼓励持续运动
    """
    
    # 目标参数
    TARGET_VELOCITY = 2    # 目标前进速度（m/s）
    DESIRED_HEIGHT = 0.3      # 标称站立高度（m）
    
    # 观测分量解析
    base_height = obs[0]
    base_pitch = obs[1]
    lin_vel_x = obs[2]
    lin_vel_y = obs[3]
    ang_vel_pitch = obs[4]
    
    # ---- 核心奖励项 ----
    # 1. 速度跟踪奖励（高斯型，0.6-1.0m/s时奖励较高）
    velocity_reward = np.exp(-2*(lin_vel_x - TARGET_VELOCITY)**2)
    
    # 2. 稳定性惩罚（二次型）
    height_penalty = 0.5*(base_height - DESIRED_HEIGHT)**2
    pitch_penalty = 0.8*base_pitch**2 + 0.1*ang_vel_pitch**2
    lateral_penalty = 1.2*lin_vel_y**2
    
    # 3. 运动对称性奖励（前腿/后腿关节差异）
    front_leg_sym = np.exp(-0.5*(obs[5]**2 + obs[6]**2))  # 前腿角度差异
    rear_leg_sym = np.exp(-0.5*(obs[9]**2 + obs[10]**2))  # 后腿角度差异
    
    # 4. 能量效率惩罚（抑制剧烈动作）
    action_penalty = 0.002*np.sum(action**2)
    
    # 5. 生存奖励（鼓励持续运动）
    survival_bonus = 0.2
    
    # ---- 加权组合 ----
    reward = (
        2.0 * velocity_reward
        - 1.5 * height_penalty
        - 1.0 * pitch_penalty
        - 0.8 * lateral_penalty
        + 0.6 * (front_leg_sym + rear_leg_sym)
        - action_penalty
        + survival_bonus
    )
    
    return reward  

def Anymal_straight_walk_reward(obs, action):
    # 提取状态变量
    x, y, z = obs[:3]  # 位置
    vx, vy, vz = obs[3:6]  # 线速度
    roll, pitch, yaw = obs[6:9]  # 欧拉角
    foot_forces = obs[9:]  # 足端力（12 维）

    # 前进奖励（目标速度 2.0 m/s）
    target_vx = 2.0
    forward_reward = np.exp(-0.5 * (vx - target_vx) ** 2)

    # 稳定性惩罚
    lateral_penalty = -vy**2 - vz**2
    roll_penalty = -roll**2
    pitch_penalty = -pitch**2

    # # 足端受力（只考虑接触脚）
    # contact_forces = foot_forces[foot_forces > 10.0]  # 选出接触脚的力
    # foot_force_penalty = -np.std(contact_forces) if len(contact_forces) > 0 else -1.0  # 腾空惩罚

    # 躯干高度（针对 ANYmal B）
    ideal_height = 0.4  # 参考 ANYmal B 正常行走的高度
    height_penalty = -abs(z - ideal_height) if z > 0.3 else -10.0


    # 计算能量消耗（假设 q_dot 不在 obs 中，所以去掉能量惩罚）

    survival_bonus = 0.2
    # 总奖励
    reward = (
        4.0 * forward_reward +
        0.5 * lateral_penalty +
        1 * (roll_penalty + pitch_penalty) +
        # 1.2 * foot_force_penalty +
        1.5 * height_penalty + survival_bonus
    )

    # 致命状态检测
    # if z < 0.3 or abs(roll) > 0.6 or abs(pitch) > 0.6:
    #     reward -= 100.0

    return reward
