
import json
import os

def view_jsonl_file(file_path, num_samples=10):
    """查看JSONL文件内容"""

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print("💡 可能的文件名:")
        current_dir = os.getcwd()
        jsonl_files = [f for f in os.listdir(current_dir) if f.endswith('.jsonl')]
        for f in jsonl_files:
            print(f"   - {f}")
        return

    print(f"📄 正在查看文件: {file_path}")
    print("=" * 60)

    try:
        # 读取文件
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # 跳过空行
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  第 {line_num} 行JSON格式错误: {e}")
                        print(f"   内容: {line.strip()[:100]}...")

        # 基本统计信息
        print(f"📊 文件统计信息:")
        print(f"   总样本数: {len(data)}")

        if data:
            # 分析数据结构
            first_item = data[0]
            print(f"   数据字段: {list(first_item.keys())}")

            # 统计不同类型的指令
            instructions = [item.get('instruction', '') for item in data]
            unique_instructions = len(set(instructions))
            print(f"   唯一指令数: {unique_instructions}")

            # 统计来源翼型
            if 'source_airfoil' in first_item:
                sources = [item.get('source_airfoil', '') for item in data]
                unique_sources = len(set(sources))
                print(f"   来源翼型数: {unique_sources}")

        print("\n" + "=" * 60)

        # 显示样本
        display_count = min(num_samples, len(data))
        print(f"📋 显示前 {display_count} 个样本:")
        print("=" * 60)

        for i in range(display_count):
            item = data[i]
            print(f"\n样本 { i +1}:")
            print(f"指令: {item.get('instruction', 'N/A')}")
            print(f"输出: {item.get('output', 'N/A')}")
            if 'source_airfoil' in item:
                print(f"来源: {item.get('source_airfoil', 'N/A')}")
            print("-" * 40)

        # 如果还有更多数据
        if len(data) > num_samples:
            print(f"\n... 还有 {len(data) - num_samples} 个样本")
            print(f"💡 想看更多？运行: view_jsonl_file('{file_path}', {len(data)})")

    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")

def analyze_output_format(file_path):
    """分析输出格式的分布"""

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"\n🔍 分析输出格式...")

    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    output = item.get('output', '')
                    outputs.append(output)
                except:
                    continue

    # 尝试解析输出中的参数
    cl_values = []
    cd_values = []
    thickness_values = []

    for output in outputs[:10]:  # 只分析前10个
        try:
            params = json.loads(output)
            if 'cl' in params:
                cl_values.append(params['cl'])
            if 'cd' in params:
                cd_values.append(params['cd'])
            if 'thickness' in params:
                thickness_values.append(params['thickness'])
        except:
            continue

    if cl_values:
        print(f"CL值范围: {min(cl_values):.3f} ~ {max(cl_values):.3f}")
    if cd_values:
        print(f"CD值范围: {min(cd_values):.4f} ~ {max(cd_values):.4f}")
    if cd_values:
        print(f"Thickness值范围: {min(thickness_values):.3f} ~ {max(thickness_values):.3f}")




if __name__ == "__main__":
    # 查看训练数据文件
    jsonl_file = "airfoil_train_mixed.jsonl"

    print("🔍 JSONL训练数据查看器")
    print("=" * 60)

    # 主要查看功能
    view_jsonl_file(jsonl_file, num_samples=5)

    # 详细分析
    analyze_output_format(jsonl_file)

    print("\n" + "=" * 60)
    print("💡 使用说明:")
    print("   - 如果要查看更多样本，修改 num_samples 参数")
    print("   - 如果文件名不同，修改 jsonl_file 变量")
    print("   - 这个脚本只是查看数据，不会修改任何文件")

