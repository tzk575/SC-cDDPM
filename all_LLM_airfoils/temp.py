import json
import random


response = ''' {
  "cl": 0.00,
  "cd": 0.004,
  "camber": 0.005,
  "thickness": 0.11
}'''




response_str = response.strip()


try:
    params = json.loads(response_str)

    cl = params["cl"]
    cd = params["cd"]
    camber = params["camber"]
    thickness = params["thickness"]

except Exception as e:
    raise ValueError(f"非法模型输出，无法解析 JSON: {response_str}")


print(cl, cd, camber, thickness)
