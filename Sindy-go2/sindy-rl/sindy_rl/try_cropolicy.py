import numpy as np
from sindy_rl.registry import Go2Sim ,AnymalSim # 根据实际路径调整
from sindy_rl.policy import RandomPolicy, CrocoddylPolicy

# 初始化环境（带渲染）
#env = Go2Sim(render_mode="human")
env = AnymalSim(render_mode="human")

# 获取动作空间参数
action_dim = env.action_space.shape[0]  # 通常为12（每条腿3个关节）
action_low = env.action_space.low[0]    # 假设动作空间对称
action_high = env.action_space.high[0]

# 重置环境
obs = env.reset()
done = False

# 安全运行参数配置
MAX_STEPS = 100000        # 最大运行步数
TILT_THRESHOLD = 0.8    # 基座倾斜角安全阈值（根据实际单位调整）
Z_HEIGHT_MIN = 0.2      # 最低允许高度（米）
a = CrocoddylPolicy()

try:
    step = 0
    while step < MAX_STEPS:
        # 生成随机动作（均匀分布）
        # a = Crocoddyl_policy
        action = a.compute_action(obs)
        
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1

except KeyboardInterrupt:
    print("\n用户手动终止操作！")
finally:
    env.close()
    print("环境已安全关闭")

# 运行结果报告
if done:
    print(f"因安全限制终止于第{step}步")
else:
    print(f"正常完成{MAX_STEPS}步测试")