import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import Env, spaces
from typing import Optional
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv


from scipy.spatial.transform import Rotation as R

class DMCEnvWrapper(DMCEnv):
    '''
    A wrapper for all dm-control environments using RLlib's 
    DMCEnv wrapper. 
    '''
    # need to wrap with config dict instead of just passing kwargs
    def __init__(self, config=None):
        env_config = config or {}
        super().__init__(**env_config)

import numpy as np
import mujoco
from gymnasium import Env, spaces
import mujoco.viewer

class Go2Sim(Env):
    def __init__(self, render_mode: Optional[str] = None):
        # 加载 MuJoCo 模型
        model_path = "go2/scene.xml"
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")
        
        self.data = mujoco.MjData(self.model)
        self.viewer = None  # 初始化渲染窗口
        self.render_mode = render_mode  # 存储渲染模式

        self.model.opt.timestep = 1 / 240  # 设置仿真时间步长
        self.target = None  # 目标初始化

        # 关节 & 速度信息
        self.n_j = self.model.nu
        self.n_q = self.model.nq
        self.n_v = self.model.nv

        print(f"n_q: {self.n_q}, n_v: {self.n_v}, n_j: {self.n_j}")

        # 设置观察空间
        obs_low = -np.inf * np.ones(13)
        obs_high = np.inf * np.ones(13)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 设置动作空间
        if self.model.actuator_ctrlrange is not None:
            act_low = self.model.actuator_ctrlrange[:, 0]
            act_high = self.model.actuator_ctrlrange[:, 1]
        else:
            act_low = -np.ones(self.n_j)
            act_high = np.ones(self.n_j)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif mode == "rgb_array":
            width, height = 480, 480
            img = np.zeros((height, width, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrContext(self.model), img)
            return img
        else:
            raise NotImplementedError("Unsupported render mode. Use 'human' or 'rgb_array'.")

        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 重置仿真数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始关节位置和速度
        q_init = np.zeros(self.n_q)
        dq_init = np.zeros(self.n_v)
        self.data.qpos[:] = q_init
        self.data.qvel[:] = dq_init
        mujoco.mj_forward(self.model, self.data)
        
        # 获取初始观察值
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 应用控制指令
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        '''
        # 更新渲染（如果需要可视化）
        if self.viewer is not None:
            self.viewer.sync()
        '''
        # 返回观察值、奖励、终止标志、截断标志和信息
        obs = self._get_obs()
        #print("Step observation:", obs)
        reward = self._get_reward(obs)
        terminated = self._get_done()  # 任务是否因失败而终止
        truncated = False  # 任务是否因时间限制而终止
        info = self._get_info()
        #self.step_counter += 1
        
        return obs, reward, terminated, truncated, info
    '''
    def _get_obs(self):
        # 读取 MuJoCo 的状态
        q = self.data.qpos.copy().astype(np.float32)  # 关节角度 + 可能的基座状态
        dq = self.data.qvel.copy().astype(np.float32)  # 关节速度 + 可能的基座速度
        
        # 处理基座状态（如果基座可以移动）
        base_position = q[:3]  # 基座位置 (x, y, z)
        base_orientation = q[3:7]  # 基座姿态（四元数）

        q_joints = q[7:]  # 关节角度

        base_linear_velocity = dq[:3]  # 基座线速度
        base_angular_velocity = dq[3:6]  # 基座角速度
        dq_joints = dq[6:]  # 关节角速度

        # 额外环境信息
        foot_contact = self.get_foot_contacts()  # 获取足端接触信息，返回形状 (4,)
        external_forces = self.get_external_forces()  # 获取外部力信息，形状 (3,)
        joint_torques = self.data.qfrc_actuator.copy().astype(np.float32)  # 关节执行器力矩
        
        # 目标信息（如果有目标）
        if self.target is not None:
            goal_info = self.get_goal_info()  # 目标位置或目标状态
        else:
            goal_info = np.array([])  # 没有目标时为空
        
        # 组合所有状态
        obs = np.concatenate([
            base_position, base_orientation,  # 机器人基座信息
            q_joints, dq_joints,  # 机器人关节状态
            base_linear_velocity, base_angular_velocity,  # 机器人运动状态
            foot_contact, external_forces, joint_torques,  # 触地信息 & 外部力 & 关节力矩
            goal_info  # 目标信息
        ])

        #原本的有62维
        return obs 
    '''
    def _get_obs(self):
        q = self.data.qpos.copy().astype(np.float32)
        dq = self.data.qvel.copy().astype(np.float32)

        # --- 核心运动学特征（9维）---
        # 基座姿态 (2维)
        base_height = q[2:3]  # 机器人基座的高度 [1]
        base_orientation = R.from_quat(q[3:7]).as_euler('xyz')[:1]  # 机器人基座的俯仰角 [1]
        

        # 基座运动 (3维)
        base_lin_velocity = dq[:2]  # 机器人基座在 X/Y 方向的线速度 [2]
        base_ang_velocity = dq[5:6]  # 机器人基座的俯仰角速度 [1]
      

        # --- 关节运动对称编码（8维）---
        # 前腿运动模式 (4维)
        front_angle_diff = q[7:9] - q[9:11]  # 前腿左右关节角度差 [2]
        front_vel_mean = (dq[6:8] + dq[8:10]) / 2  # 前腿对称平均速度 [2]
        

        # 后腿运动模式 (4维)
        rear_angle_diff = q[11:13] - q[13:15]  # 后腿左右关节角度差 [2]
        rear_vel_mean = (dq[10:12] + dq[12:14]) / 2  # 后腿平均速度 [2]
       

        # --- 组合观测（总维度：2+3+4+4=13）---
        obs = np.concatenate([
            base_height,  # 机器人基座的高度 [1]
            base_orientation,  # 机器人基座的俯仰角 [1]
            base_lin_velocity,  # 机器人基座的 X/Y 方向线速度 [2]
            base_ang_velocity,  # 机器人基座的俯仰角速度 [1]
            front_angle_diff,  # 前腿左右关节角度差 [2]
            front_vel_mean,  # 前腿对称平均速度 [2]
            rear_angle_diff,  # 后腿左右关节角度差 [2]
            rear_vel_mean,  # 后腿平均速度 [2]
        ])
        return obs
        


    def get_foot_contacts(self):
        """获取足端接触信息，返回形状 (4,)，表示四个足端是否接触地面"""
        contact_flags = np.zeros(4, dtype=np.float32)  # 4 个足端

        # 获取几何体名称列表
        geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]

        # 使用 self.data.ncon 获取当前时间步的实际接触点数
        for contact in self.data.contact[: self.data.ncon]:
            # 获取几何体 ID 对应的名称
            geom1_name = geom_names[contact.geom1]
            geom2_name = geom_names[contact.geom2]

            # 检查是否是足端接触地面（假设足端几何体命名为 foot_0, foot_1, foot_2, foot_3）
            for i in range(4):
                if f"foot_{i}" in (geom1_name, geom2_name):  
                    contact_flags[i] = 1.0  # 设为 1，表示接触地面

        return contact_flags  # 返回四个足端的接触状态（0 或 1）

    def get_external_forces(self):
        """获取基座受到的外部力，返回形状 (3,)，表示 x, y, z 方向的外力"""
        base_force = self.data.cfrc_ext[0, :3]  # 机器人基座的外力（前三个分量）
        return base_force.astype(np.float32)

    def _get_reward(self,obs):
            # 定义奖励函数（根据任务需求）
                # 奖励系数配置（可调整）
        FORWARD_WEIGHT = 2.0      # 前进速度奖励
        LATERAL_PENALTY = 0.5     # 横向移动惩罚
        PITCH_PENALTY = 0.3       # 俯仰角惩罚
        HEIGHT_PENALTY = 0.4      # 基座高度惩罚（新增）
        ACT = 0.01  # 动作平滑性惩罚
        SYMMETRY_PENALTY = 0.2    # 对称运动惩罚

        # 观测值解析（严格对齐观测空间结构）
        base_height = obs[0]                   # 基座高度 [1]
        pitch = obs[1]                         # 俯仰角 [1]
        lin_vel_x, lin_vel_y = obs[2], obs[3]   # 线速度XY [2]
        front_angle_diff = obs[5:7]            # 前腿关节角度差 [2]
        rear_angle_diff = obs[9:11]            # 后腿关节角度差 [2]

        # 核心奖励项 --------------------------------------------------
        # 1. 前进速度奖励（正向线性激励）
        forward_reward = FORWARD_WEIGHT * lin_vel_x
        
        # 2. 横向稳定性惩罚（抑制侧滑）
        lateral_penalty = LATERAL_PENALTY * (lin_vel_y ** 2)
        
        # 3. 姿态稳定性惩罚（保持水平姿态）
        pitch_cost = PITCH_PENALTY * (pitch ** 2)
        
        # 4. 基座高度惩罚（新增，维持目标高度）
        height_cost = HEIGHT_PENALTY * (base_height - 0.3) ** 2  # 假设目标高度0.3m
        
        # 5. 对称性惩罚（抑制肢体运动不对称）
        symmetry_penalty = SYMMETRY_PENALTY * (
            np.sum(front_angle_diff ** 2) + np.sum(rear_angle_diff ** 2)
        )
        
        # 6. 动作平滑性惩罚（抑制突变控制）
        #action_penalty = ACT * np.sum(np.square(action))

        # 总奖励计算 --------------------------------------------------
        total_reward = (
            forward_reward
            - lateral_penalty
            - pitch_cost
            - height_cost
            - symmetry_penalty
            #- action_penalty
        )
        
        return total_reward
        

    def _get_done(self):
        # 定义终止条件（根据任务需求）
        return False

    def _get_info(self):
        # 返回额外的信息（可选）
        return {}
    '''
    def close(self):
        # 关闭渲染器
        if self.viewer is not None:
            self.viewer.close()
    '''
    
class AnymalSim(Env):
    def __init__(self, render_mode: Optional[str] = None):

        self.model = mujoco.MjModel.from_xml_path("/root/projects/anybotics_anymal_b/scene.xml")  
        self.data = mujoco.MjData(self.model)

        # with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
        #     while viewer.is_running():
        #         mujoco.mj_step(self.model, self.data)

       
        # 初始化模拟数据和其他配置
        self.viewer = None
        self.render_mode = render_mode
        self.model.opt.timestep = 1 / 240  # 设置时间步长
        self.target = None
        
        # 获取机器人的关节数、位置数和速度数
        self.n_j = self.model.nu
        self.n_q = self.model.nq
        self.n_v = self.model.nv
        
        # print(f"n_q: {self.n_q}, n_v: {self.n_v}, n_j: {self.n_j}")
        
        # 设置观测空间的上下限
        obs_low = -np.inf * np.ones(20)  # 根据ANYmal的状态空间大小调整
        obs_high = np.inf * np.ones(20)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        # 设置控制动作空间的上下限
        act_low = -40  # 根据需要调整动作的限制
        act_high = 40
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
    
    def render(self, mode="human"):
        # 渲染模式，根据要求选择'视觉'或'rgb_array'
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif mode == "rgb_array":
            width, height = 480, 480
            img = np.zeros((height, width, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrContext(self.model), img)
            return img
        else:
            raise NotImplementedError("不支持的渲染模式。请选择 'human' 或 'rgb_array'。")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 重置仿真数据
        mujoco.mj_resetData(self.model, self.data)

        # 设置ANYmal机器人的初始关节位置和速度
        q_init = np.zeros(self.n_q)
        dq_init = np.zeros(self.n_v)

        # 设置合理的初始位置，避免掉到地面（可根据需要调整）
        q_init[2] = 0.5  # 设置z轴初始位置

        # 设置默认的站立姿态（如果ANYmal的四元数旋转值不同，请根据需要调整）
        q_init[3:7] = np.array([1, 0, 0, 0])  # 这是单位四元数，表示没有旋转

        # 将初始状态应用到模拟器中
        self.data.qpos[:] = q_init
        self.data.qvel[:] = dq_init

        # 进行一步仿真，以确保状态稳定
        mujoco.mj_forward(self.model, self.data)

        # 获取初始观察值
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        # 将控制命令应用到模拟器
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # 获取更新后的观察值，计算奖励，判断任务是否完成等
        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = self._get_done()  # 判断任务是否因为失败而结束
        truncated = False  # 判断任务是否因为时间限制而结束
        info = self._get_info()

        return obs, reward, terminated, truncated, info
    
    def _get_info(self):
        # 返回额外信息（如果需要的话）
        return {}

    def _get_reward(self, obs, action):
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

        # 足端受力（只考虑接触脚）
        contact_forces = foot_forces[foot_forces > 10.0]  # 选出接触脚的力
        foot_force_penalty = -np.std(contact_forces) if len(contact_forces) > 0 else -1.0  # 腾空惩罚

        # 躯干高度（针对 ANYmal B）
        ideal_height = 0.4  # 参考 ANYmal B 正常行走的高度
        height_penalty = -abs(z - ideal_height) if z > 0.3 else -10.0


        # 计算能量消耗（假设 q_dot 不在 obs 中，所以去掉能量惩罚）

        # 总奖励
        reward = (
            3.0 * forward_reward +
            1.5 * lateral_penalty +
            2.0 * (roll_penalty + pitch_penalty) +
            1.2 * foot_force_penalty +
            1.5 * height_penalty
        )

        # 致命状态检测
        # if z < 0.3 or abs(roll) > 0.6 or abs(pitch) > 0.6:
        #     reward -= 100.0

        return reward
    
    def _get_done(self):
        # 判断任务是否结束（例如机器人翻倒或失败）
        return False

    def _get_obs(self):
        # 获取当前的关节位置和速度，并构造观察值
        q = self.data.qpos.copy().astype(np.float32)
        dq = self.data.qvel.copy().astype(np.float32)

        # 将四元数转换为欧拉角
        euler_angles = R.from_quat(q[3:7]).as_euler('xyz')  # ϕ, θ, ψ 欧拉角

        # # 计算每个欧拉角的余弦和正弦值
        # cos_phi, sin_phi = np.cos(euler_angles[0]), np.sin(euler_angles[0])
        # cos_theta, sin_theta = np.cos(euler_angles[1]), np.sin(euler_angles[1])
        # cos_psi, sin_psi = np.cos(euler_angles[2]), np.sin(euler_angles[2])

        # 获取传感器数据
        self.LF_foot_force = self.data.sensor('LF_foot_force').data
        self.RF_foot_force = self.data.sensor('RF_foot_force').data
        self.LH_foot_force = self.data.sensor('LH_foot_force').data
        self.RH_foot_force = self.data.sensor('RH_foot_force').data

        # 构造观察向量（根据ANYmal的状态进行调整）
        X = np.concatenate([
            q[:3],  # x, y, z 位置
            dq[:3],  # vx, vy, vz 线速度
            # [cos_phi, sin_phi, cos_theta, sin_theta, cos_psi, sin_psi],  # 欧拉角的余弦和正弦值
            euler_angles,#3维
            self.LF_foot_force,self.RF_foot_force,self.LH_foot_force,self.RH_foot_force      # 足端接触状态（力）12维
        ])
        # print("obs",X)
    
        return X
    
    def get_foot_contacts(self):
        # 检测足端接触（根据ANYmal模型的足端命名进行调整）
        contact_flags = np.zeros(4, dtype=np.float32)
        geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        for contact in self.data.contact[: self.data.ncon]:
            geom1_name = geom_names[contact.geom1]
            geom2_name = geom_names[contact.geom2]
            for i in range(4):
                if f"foot_{i}" in (geom1_name, geom2_name):  
                    contact_flags[i] = 1.0
        return contact_flags


class Go2wrapper(Go2Sim):
    def __init__(self, config=None):
        super().__init__(**(config or {}))  # 如果 config=None，则用空字典 {}

class Anymalwrapper(AnymalSim):
    def __init__(self, config=None):
        super().__init__(**(config or {}))  # 如果 config=None，则用空字典 {}