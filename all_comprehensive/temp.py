
from LLM.airfoil_LLM import AirfoilLLMTester
import numpy as np


def jitter_params(
    params: dict,
    batch: int = 16,
    enable_jitter: bool = True,
    jitter_ratio: dict = None,
    fixed_params: list = None,
):
    """
    根据 LLM 给出的参数，生成 batch 份 numpy 条件向量
    """

    if jitter_ratio is None:
        jitter_ratio = {
            "cl": 0.05,
            "cd": 0.03,
            "thickness": 0.08,
            "camber": 0.0,      # camber 默认不抖
        }

    if fixed_params is None:
        fixed_params = ["camber"]

    # 初始化数组
    cl = np.full(batch, params["cl"], dtype=np.float32)
    cd = np.full(batch, params["cd"], dtype=np.float32)
    camber = np.full(batch, params["camber"], dtype=np.float32)
    thickness = np.full(batch, params["thickness"], dtype=np.float32)

    if enable_jitter:
        for name, arr in zip(
            ["cl", "cd", "camber", "thickness"],
            [cl, cd, camber, thickness],
        ):
            if name in fixed_params:
                continue

            ratio = jitter_ratio.get(name, 0.0)
            noise = np.random.uniform(
                low=-ratio,
                high=ratio,
                size=batch
            )
            arr *= (1.0 + noise)

    # 组合 condition
    condition = np.stack(
        [cl, cd, camber, thickness],
        axis=1
    )

    return cl, cd, camber, thickness, condition




airfoil_LLM = AirfoilLLMTester()


test_prompt1 = '''
    指令: 设计目标：
    设计一个对称翼型。
    
    初始设计参数：
    cl = -0.0
    cd = 0.00535
    camber = 0.0126
    thickness = 0.121
    
    任务说明：
    请确认或微调上述参数，使其适合用于后续条件扩散模型的几何生成。
    '''


params, embedding = airfoil_LLM.generate_airfoil_params(test_prompt1)

print(f"📥 输入 Prompt:")
print(test_prompt1.strip())

print('-----------' * 3)
print(params)
print(type(params))
print(params.keys())


print('====' * 5)


batch_size = 16
# 改成 False 就是“重复”
cl, cd, camber, thickness, condition = jitter_params(params, batch_size, enable_jitter=True)




