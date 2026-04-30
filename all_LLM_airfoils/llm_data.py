

import os
import pickle
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import aerosandbox as asb



# 设置字体 - 修复负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置英文字体为 Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class AirfoilDataset(Dataset):
    def __init__(self, airfoil_path, num_points_per_side=100, cache_file='airfoil_cache.pkl'):
        self.airfoil_path = airfoil_path
        self.airfoil_files = os.listdir(airfoil_path)
        self.airfoil_files = [f for f in self.airfoil_files if f.endswith(".dat")]
        self.num_points_per_side = num_points_per_side
        self.cache_file = cache_file

        if os.path.exists(self.cache_file):
            # Load cached data
            with open(self.cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.coordinates = cache['coordinates']
                self.diffusion_training_coordinates = cache['diffusion_training_coordinates']
                self.CD = cache['CD']
                self.CL = cache['CL']
                self.CM = cache['CM']
                self.max_camber = cache['max_camber']
                self.max_thickness = cache['max_thickness']
                self.TE_thickness = cache['TE_thickness']
                self.TE_angle = cache['TE_angle']
                self.names = cache['names']
        else:
            # Convert airfoils to airfoil objects
            self.airfoils = []
            for airfoil_file in self.airfoil_files:
                filepath = os.path.join(airfoil_path, airfoil_file)
                airfoil_name = airfoil_file.split(".")[0]
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    cleaned_lines = [line.strip() for line in lines if line.strip()]

                    # Remove header line if present
                    if not cleaned_lines[0][0].isdigit():
                        cleaned_lines = cleaned_lines[1:]

                    seen_lines = set()
                    unique_lines = []
                    # Remove duplicate points except for the first and last points
                    for i, line in enumerate(cleaned_lines):
                        if i == 0 or i == (len(cleaned_lines) - 1):
                            unique_lines.append(line)
                            seen_lines.add(line)
                        elif line not in seen_lines:
                            seen_lines.add(line)
                            unique_lines.append(line)
                    if len(cleaned_lines) != len(unique_lines):
                        print(f"Airfoil {airfoil_name} had duplicate points. Duplicate points were removed.")
                        with open(filepath, "w") as f:
                            print(f"Writing cleaned airfoil to {filepath}")
                            f.write("\n".join(unique_lines) + "\n")
                # Convert airfoil to airfoil object
                try:
                    airfoil = asb.Airfoil(
                        name=airfoil_name,
                        coordinates=filepath
                    )
                    self.airfoils.append(airfoil)
                except Exception as e:
                    print(f"Error loading airfoil {airfoil_name}: {e}")

            # Repanelize all airfoils
            self.repanelized_airfoils = []
            for airfoil in self.airfoils:
                try:
                    if len(airfoil.coordinates) < 2:
                        print(f"Skipping airfoil {airfoil.name} due to insufficient coordinates.")
                        continue
                    repanelized_airfoil = airfoil.repanel(n_points_per_side=num_points_per_side)
                    self.repanelized_airfoils.append(repanelized_airfoil)
                except Exception as e:
                    print(f"Error repaneling airfoil {airfoil.name}: {e}")

            # Assuming self.repanelized_airfoils is already defined and contains airfoils with coordinates
            self.coordinates = [airfoil.coordinates for airfoil in self.repanelized_airfoils]
            self.upper_coord = [airfoil.upper_coordinates() for airfoil in self.repanelized_airfoils]
            self.lower_coord = [airfoil.lower_coordinates() for airfoil in self.repanelized_airfoils]
            self.diffusion_training_coordinates = [np.vstack((upper, lower)) for upper, lower in
                                                   zip(self.upper_coord, self.lower_coord)]

            self.CD = []
            self.CL = []
            self.CM = []
            self.max_camber = []
            self.max_thickness = []
            self.TE_thickness = []
            self.TE_angle = []
            self.names = []
            for airfoil in self.repanelized_airfoils:
                print(f"Calculating CD and CL for airfoil {airfoil.name}")
                self.names.append(airfoil.name)
                coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
                print(f"CL: {coef['CL'][0]}, CD: {coef['CD'][0]}")
                max_camber = airfoil.max_camber()
                max_thickness = airfoil.max_thickness()
                TE_thickness = airfoil.TE_thickness()
                TE_angle = airfoil.TE_angle()
                self.CL.append(coef['CL'][0])
                self.CD.append(coef['CD'][0])
                self.CM.append(coef['CM'][0])
                self.max_camber.append(max_camber)
                self.max_thickness.append(max_thickness)
                self.TE_thickness.append(TE_thickness)
                self.TE_angle.append(TE_angle)

            # Save to cache
            cache = {
                'coordinates': self.coordinates,
                'diffusion_training_coordinates': self.diffusion_training_coordinates,
                'CD': self.CD,
                'CL': self.CL,
                'CM': self.CM,
                'max_camber': self.max_camber,
                'max_thickness': self.max_thickness,
                'TE_thickness': self.TE_thickness,
                'TE_angle': self.TE_angle,
                'names': self.names
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)

        if os.path.exists('llm_training_data.pkl'):
            with open('llm_training_data.pkl', 'rb') as f:
                self.llm_training_data = pickle.load(f)
        else:
            self.llm_training_data = self._build_llm_training_data()
            with open('llm_training_data.pkl', 'wb') as f:
                pickle.dump(self.llm_training_data, f)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        train_coords = self.diffusion_training_coordinates[idx]
        train_coords_y = train_coords[:, 1]  # only pass the y coordinates to the model
        cd = self.CD[idx]
        cl = self.CL[idx]
        cm = self.CM[idx]
        max_camber = self.max_camber[idx]
        max_thickness = self.max_thickness[idx]
        TE_thickness = self.TE_thickness[idx]
        TE_angle = self.TE_angle[idx]
        name = self.names[idx]

        # Separate the y-coordinates into two parts
        train_coords_y_upper = train_coords_y[:self.num_points_per_side]  # First 100 points
        train_coords_y_lower = train_coords_y[self.num_points_per_side:]  # Second 100 points

        # Stack them to create a 2-channel tensor
        train_coords_y = np.stack([train_coords_y_upper, train_coords_y_lower], axis=0)

        # Convert data to the required shape: (channels, data)
        coordinates = coordinates.T  # Transpose to shape (2, data_points)

        return {
            'train_coords_y': torch.tensor(train_coords_y, dtype=torch.float32),
            'coordinates': torch.tensor(coordinates, dtype=torch.float32),
            'CD': cd,
            'CL': cl,
            'CM': cm,
            'max_camber': max_camber,
            'max_thickness': max_thickness,
            'TE_thickness': TE_thickness,
            'TE_angle': TE_angle,
            'name': name
        }

    def get_x(self):
        return self.diffusion_training_coordinates[0][:, 0]

    def _build_llm_training_data(self):
        """基于现有翼型参数反向构建LLM训练数据"""
        training_data = []

        for i in range(len(self.names)):
            # 提取参数
            params = {
                'cl': self.CL[i],
                'cd': self.CD[i],
                'camber': self.max_camber[i],
                'thickness': self.max_thickness[i],
                'name': self.names[i]
            }

            # 生成多个描述变体
            descriptions = self._generate_descriptions(params)

            # 参数输出格式
            param_json = json.dumps({
                "cl": round(params['cl'], 3),
                "cd": round(params['cd'], 4),
                "camber": round(params['camber'], 3),
                "thickness": round(params['thickness'], 3)
            })

            # 为每个翼型生成多条训练样本
            for desc in descriptions:
                training_data.append({
                    "instruction": desc,
                    "output": param_json,
                    "source_airfoil": params['name']
                })

        return training_data

    def _generate_descriptions(self, params):
        """根据翼型参数生成自然语言描述"""
        cl, cd, camber, thickness = params['cl'], params['cd'], params['camber'], params['thickness']

        descriptions = []

        # 基于性能特征的描述
        if cl > 1.0 and cd < 0.01:
            descriptions.extend([
                "设计一个高升力低阻力的翼型",
                "需要升力强劲且阻力较小的翼型",
                "起降性能优秀的高效翼型"
            ])
        elif cl > 0.6 and cd < 0.008:
            descriptions.extend([
                "设计一个高效率的翼型",
                "无人机长航时翼型",
                "追求升阻比的优化翼型"
            ])
        elif cl < 0.4:
            descriptions.extend([
                "设计一个高速飞行翼型",
                "超音速或跨音速翼型",
                "低升力薄翼型设计"
            ])
        else:
            descriptions.extend([
                "设计一个平衡性能的翼型",
                "通用航空翼型",
                "常规巡航翼型"
            ])

        # 基于几何特征的描述
        if thickness < 0.08:
            descriptions.extend([
                "设计一个薄翼型",
                "厚度较小的翼型设计",
                "适合高速的薄剖面翼型"
            ])
        elif thickness > 0.15:
            descriptions.extend([
                "设计一个厚翼型",
                "结构强度要求高的翼型",
                "厚度较大的翼型"
            ])

        if camber > 0.05:
            descriptions.extend([
                "设计一个大弯度翼型",
                "高弯度高升力翼型"
            ])
        elif camber < 0.01:
            descriptions.extend([
                "设计一个对称翼型",
                "零弯度或近似对称的翼型"
            ])

        # 基于应用场景的描述
        if cl > 0.7 and cd < 0.008:
            descriptions.extend([
                "无人机用高效翼型",
                "长航时翼型设计"
            ])
        if cl < 0.5 and thickness < 0.1:
            descriptions.extend([
                "战斗机用翼型",
                "高速机动翼型"
            ])
        if 0.4 < cl < 0.7 and 0.007 < cd < 0.01:
            descriptions.extend([
                "商用客机翼型",
                "民航运输机翼型"
            ])

        return descriptions[:4]  # 返回4个变体，避免过多重复

    def get_llm_training_data(self):
        """获取LLM训练数据"""
        return self.llm_training_data


# Usage example
# 生成LLM微调数据
if __name__ == '__main__':
    dataset = AirfoilDataset('coord_seligFmt')
    llm_data = dataset.get_llm_training_data()

    print(f"生成了 {len(llm_data)} 条LLM训练数据")



    # 保存为JSONL格式（微调常用格式）
    with open('airfoil_llm_training.jsonl', 'w', encoding='utf-8') as f:
        for item in llm_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 查看数据样例
    for i in range(10):
        print(f"样例 {i + 1}:")

        print(f"来源: {llm_data[i]}")
        print("-" * 50)
