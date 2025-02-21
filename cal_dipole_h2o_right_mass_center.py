import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

def read_bader_file(filename):
    """
    读取Bader分析输出文件，返回所有必要信息
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # 读取第二行的信息
        second_line = lines[1].split()
        grid_dims = list(map(int, second_line[:3]))  # nx, ny, nz
        n_atoms = int(second_line[6])  # 原子总数
        n_elements = int(second_line[7])  # 元素种类数
        print(f"\n网格维度: {grid_dims}")
        print(f"原子总数: {n_atoms}")
        print(f"元素种类数: {n_elements}")
        
        # 跳过第三行（包含一些额外参数）
        # 读取晶格向量（第4-6行）
        cell = np.zeros((3, 3))
        for i in range(3):
            cell[i] = list(map(float, lines[3+i].split()[:3]))
        print("\n晶格向量:")
        print(cell)
            
        # 计算网格间距
        dx = cell[0][0] / grid_dims[0]
        dy = cell[1][1] / grid_dims[1]
        dz = cell[2][2] / grid_dims[2]
        
        # 跳过第7行
        # 读取原子类型和电荷映射（从第8行开始）
        type_to_charge = {}
        type_to_element = {}  # 新增：存储类型到元素符号的映射
        current_line = 7
        print("\n原子类型和电荷映射:")
        for i in range(n_elements):
            parts = lines[current_line+i].split()
            type_index = int(parts[0])
            element = parts[1]  # 这是元素符号
            charge = float(parts[2])
            type_to_charge[type_index] = charge
            type_to_element[type_index] = element  # 保存元素符号
            print(f"类型 {type_index}: {element} (电荷: {charge})")
        
        # 读取原子坐标和类型
        coords = []
        charges = {}
        elements = []  # 新增：存储每个原子的元素符号
        current_line += n_elements
        print("\n原子坐标和类型:")
        for i in range(n_atoms):
            parts = lines[current_line+i].split()
            atom_index = int(parts[0])
            coord = list(map(float, parts[1:4]))
            atom_type = int(parts[4])
            coords.append(coord)
            charges[atom_index] = type_to_charge[atom_type]
            elements.append(type_to_element[atom_type])  # 保存元素符号
            print(f"原子 {atom_index}: {type_to_element[atom_type]} 坐标 {coord}, 类型 {atom_type}, 电荷 {charges[atom_index]}")
            
        # 读取电荷密度数据部分的修改
        density_start = current_line + n_atoms 
        density_data = np.zeros((grid_dims[0], grid_dims[1], grid_dims[2]))
        print(f"\n开始读取电荷密度数据，从第 {density_start} 行开始")
        
        # 计算预期的数据点数量
        expected_points = grid_dims[0] * grid_dims[1] * grid_dims[2]
        expected_lines = (expected_points + 4) // 5  # 每行5个值
        
        print(f"预期数据点数量: {expected_points}")
        print(f"预期行数: {expected_lines}")
        
        # 按照cube文件的正确顺序读取数据：z->y->x
        data_index = 0
        for z in range(grid_dims[2]):
            for y in range(grid_dims[1]):
                for x in range(grid_dims[0]):
                    try:
                        # 计算正确的索引
                        line_index, value_index = divmod(data_index, 5)
                        density_data[x,y,z] = float(lines[density_start + line_index].split()[value_index])
                        data_index += 1
                    except (IndexError, ValueError) as e:
                        raise ValueError(f"在读取第 {density_start + line_index} 行第 {value_index} 个值时出错。\n"
                                      f"当前索引: x={x}, y={y}, z={z}\n"
                                      f"数据索引: {data_index}\n"
                                      f"请检查文件格式是否正确。") from e
        
        print(f"电荷密度数组形状: {density_data.shape}")
            
        # 计算并打印总电子数
        volume_element = dx * dy * dz
        total_electrons = np.sum(density_data) * volume_element * (1.8897259886)**3
        print(f"\n总电子数: {total_electrons:.6f}")
            
        return np.array(coords), charges, density_data, (dx, dy, dz), cell, total_electrons, elements  # 添加elements到返回值

def calculate_dipole(coords, charges, density, grid_info, cell, total_electrons, elements, ref_point='cell_center'):
    """
    计算总偶极矩，包括电子和核的贡献
    考虑周期性边界条件
    
    参数:
        ref_point: 可选 'cell_center' 或 'mass_center'，指定偶极矩计算的参考点
    """
    print(f"\n系统总电子数: {total_electrons:.6f}")
    print(f"使用的参考点: {ref_point}")
    
    dx, dy, dz = grid_info
    nx, ny, nz = density.shape
    
    # 创建网格点坐标（考虑周期性边界条件）
    x = np.linspace(0, 1, nx, endpoint=False)  # 使用分数坐标
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 转换到实空间坐标
    X = X * cell[0][0] + Y * cell[1][0] + Z * cell[2][0]
    Y = X * cell[0][1] + Y * cell[1][1] + Z * cell[2][1]
    Z = X * cell[0][2] + Y * cell[1][2] + Z * cell[2][2]
    
    # 转换常数
    bohr_to_ang = 1.8897259886    # Bohr到Angstrom的转换因子
    volume_element = dx * dy * dz * bohr_to_ang**3  # 体积元素（Å³）
    
    # 确定参考点
    if ref_point == 'cell_center':
        reference_point = np.sum(cell, axis=0) / 2
        print(f"使用晶胞中心作为参考点: {reference_point}")
    elif ref_point == 'mass_center':
        # 计算质心
        masses = np.array([atomic_numbers[elem] for elem in elements])  # 获取原子质量
        total_mass = np.sum(masses)
        mass_center = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        reference_point = mass_center
        print(f"使用质心作为参考点: {reference_point}")
    else:
        raise ValueError("ref_point 必须是 'cell_center' 或 'mass_center'")
    
    # 计算电子偶极矩（与电荷密度相关的部分）
    # 相对于参考点的坐标
    X_rel = X - reference_point[0]
    Y_rel = Y - reference_point[1]
    Z_rel = Z - reference_point[2]
    
    electronic_dipole = np.zeros(3)
    electronic_dipole[0] = -np.sum(X_rel * density) * volume_element
    electronic_dipole[1] = -np.sum(Y_rel * density) * volume_element
    electronic_dipole[2] = -np.sum(Z_rel * density) * volume_element
    
    # 计算核偶极矩（考虑周期性边界条件）
    nuclear_dipole = np.zeros(3)
    
    for i, coord in enumerate(coords):
        atom_index = i + 1
        charge = charges[atom_index]
        
        # 将原子坐标转换为分数坐标
        frac_coord = np.linalg.solve(cell.T, coord)
        # 将分数坐标折叠到[0,1)区间
        frac_coord = frac_coord % 1.0
        # 转换回笛卡尔坐标
        cart_coord = np.dot(cell.T, frac_coord)
        
        # 选择最近的周期性映像
        for j in range(3):
            while cart_coord[j] - reference_point[j] > cell[j][j]/2:
                cart_coord[j] -= cell[j][j]
            while cart_coord[j] - reference_point[j] < -cell[j][j]/2:
                cart_coord[j] += cell[j][j]
        
        # 计算相对于参考点的偶极矩
        nuclear_dipole += charge * (cart_coord - reference_point)
    
    # 计算总偶极矩（单位：e·Å）
    total_dipole = nuclear_dipole + electronic_dipole
    
    # 转换到Debye（1 e·Å = 4.8032 Debye）
    debye_conversion = 4.8032
    total_dipole *= debye_conversion
    total_magnitude = np.sqrt(np.sum(total_dipole**2))
    
    print("\n电子偶极矩 (e·Å):")
    print(f"X: {electronic_dipole[0]:.6f}")
    print(f"Y: {electronic_dipole[1]:.6f}")
    print(f"Z: {electronic_dipole[2]:.6f}")
    
    print("\n核偶极矩 (e·Å):")
    print(f"X: {nuclear_dipole[0]:.6f}")
    print(f"Y: {nuclear_dipole[1]:.6f}")
    print(f"Z: {nuclear_dipole[2]:.6f}")
    
    return total_dipole, total_magnitude

def write_xyz_file(coords, elements, name):
    """
    使用实际的元素符号生成xyz格式文件
    """
    with open(f'{name}.xyz', 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write("Generated by cal_dipole.py\n")
        for coord, element in zip(coords, elements):
            f.write(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def write_vmd_script(name, dipole, n_atoms, vmd_path=r"D:\project\electrostatic potential\VMD", scale=1.0, arrow_radius=0.08, arrow_body=0.85, arrow_color='green'):
    """
    生成VMD可视化脚本和批处理文件
    """
    # 生成VMD脚本
    with open(f'view_{name}.vmd', 'w') as f:
        # 首先加载分子和设置显示样式
        f.write(f"""mol new {name}.xyz
mol modstyle 0 0 CPK 1.0 0.3 12.0 12.0
mol modcolor 0 0 Element

color Name C tan
color change rgb tan 0.700000 0.560000 0.360000
# 设置背景颜色为白色
color Display Background white
material change mirror Opaque 0.15
material change outline Opaque 4.000000
material change outlinewidth Opaque 0.2
material change ambient Glossy 0.1
material change diffuse Glossy 0.600000
material change opacity Glossy 0.75
material change ambient Opaque 0.08
material change mirror Opaque 0.0
material change shininess Glossy 1.0
mol modcolor 1 top ColorID 12
mol modcolor 2 top ColorID 22
display distance -7.0
display height 10
light 3 on


""")
        
        # 然后写入drawarrow过程定义，注意这里大括号的处理

        draw_content= f"""proc drawarrow {{atmrange fragdx fragdy fragdz {{scl 1}} {{rad {arrow_radius}}} {{showgoc 1}}}} {{
#Determine arrow center
set sel [atomselect top $atmrange]
set cen [measure center $sel]
set cenx [lindex $cen 0]
set ceny [lindex $cen 1]
set cenz [lindex $cen 2]
if {{ $showgoc==1 }} {{puts "Geometry center: $cenx $ceny $cenz"}}
#Scale vector
set fragdx [expr $fragdx*$scl]
set fragdy [expr $fragdy*$scl]
set fragdz [expr $fragdz*$scl]
#Draw arrow
set body {arrow_body}
set begx [expr $cenx-$fragdx/2]
set begy [expr $ceny-$fragdy/2]
set begz [expr $cenz-$fragdz/2]
set endx [expr $cenx+$fragdx*$body-$fragdx/2]
set endy [expr $ceny+$fragdy*$body-$fragdy/2]
set endz [expr $cenz+$fragdz*$body-$fragdz/2]
draw cylinder "$begx $begy $begz" "$endx $endy $endz" radius $rad filled yes resolution 20
set endx2 [expr $cenx+$fragdx/2]
set endy2 [expr $ceny+$fragdy/2]
set endz2 [expr $cenz+$fragdz/2]
draw cone "$endx $endy $endz" "$endx2 $endy2 $endz2" radius [expr $rad*2.5] resolution 20
}}
"""

        f.write(draw_content)
        # 最后添加绘制箭头的命令
        f.write(f"""
draw color {arrow_color}
drawarrow "serial 1 to {n_atoms}" {dipole[0]:.6f} {dipole[1]:.6f} {dipole[2]:.6f} {scale}
""")

    # 生成批处理文件
    with open(f'view_{name}.bat', 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'cd /d "%~dp0"\n')
        f.write(f'"{vmd_path}\\vmd.exe" -e "view_{name}.vmd"\n')
        f.write('pause\n')

def main():
    name = 'r-p-MBA'
    coords, charges, density, grid_info, cell, total_electrons, elements = read_bader_file(f'Bader_{name}')
    
    ### 箭头绘制参数
    arrow_length_scale = 0.9
    arrow_radius = 0.08
    arrow_body = 0.85
    arrow_color = 'red'

    # 添加参考点选项
    ref_point = 'mass_center'  # 'mass_center'或 'cell_center'
    total_dipole, total_magnitude = calculate_dipole(
        coords, charges, density, grid_info, cell, total_electrons, elements, ref_point)
    
    print("\n偶极矩结果 (Debye):")
    print(f"X方向: {total_dipole[0]:.6f}")
    print(f"Y方向: {total_dipole[1]:.6f}")
    print(f"Z方向: {total_dipole[2]:.6f}")
    print(f"总偶极矩大小: {total_magnitude:.6f}")
    
    # 输出用空格分隔的偶极矩向量
    print(f"\n偶极矩向量 (用于VMD):")
    print(f"{total_dipole[0]:.6f} {total_dipole[1]:.6f} {total_dipole[2]:.6f}")
    
    # 生成xyz文件时使用实际的元素符号
    write_xyz_file(coords, elements, name)
    
    # 生成VMD脚本和批处理文件
    write_vmd_script(name, total_dipole, len(coords), scale=arrow_length_scale, arrow_radius=arrow_radius, arrow_body=arrow_body, arrow_color=arrow_color)
    
    print(f"\n已生成以下文件：")
    print(f"1. {name}.xyz - 结构文件")
    print(f"2. view_{name}.vmd - VMD可视化脚本")
    print(f"3. view_{name}.bat - 启动脚本")
    print(f"\n双击 view_{name}.bat 即可在VMD中查看结构和偶极矩方向")

    # 计算总核电荷
    total_nuclear_charge = sum(charges.values())
    # 计算总电荷（核电荷减去电子数）
    total_charge = total_nuclear_charge - total_electrons
    print(f"系统总核电荷: {total_nuclear_charge:.6f}")
    print(f"系统总电荷: {total_charge:.6f} (应接近0如果是中性系统)")

if __name__ == "__main__":
    main()
