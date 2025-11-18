# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, sys

class PointCloudSurfaceGenerator:
    self.surfaces = []
    self.coordinate_system = {
        'origin': np.array([0, 0, 0]),
        'x_axis': np.array([1, 0, 0]),
        'y_axis': np.array([0, 1, 0]),
        'z_axis': np.array([0, 0, 1])
    }

    def generate_coordinate_system(self, point_cloud):
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        center = (min_coords + max_coords) / 2
        self.coordinate_system['origin'] = center
        self.coordinate_system['bounds'] = {'min': min_coords, 'max': max_coords, 'center': center}
        return self.coordinate_system

    def extract_surfaces(self, point_cloud, eps=0.1, min_samples=20):
        print("开始表面提取...")
        print(f"输入点云大小: {point_cloud.shape}")
        coord_system = self.generate_coordinate_system(point_cloud)
        print(f"坐标系原点: {coord_system['origin']}")

        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(point_cloud)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"找到 {n_clusters} 个聚类，噪声点数量: {n_noise}")

        self.surfaces = []
        for cluster_id in range(n_clusters):
            cluster_points = point_cloud[labels == cluster_id]
            if len(cluster_points) < 3: continue
            surface = self._fit_plane_and_calculate_equation(cluster_points)
            surface.update({'cluster_id': cluster_id,
                            'point_count': len(cluster_points),
                            'points': cluster_points})
            self.surfaces.append(surface)
        print(f"成功提取 {len(self.surfaces)} 个表面")
        return self.surfaces, labels

    # ====================== 产状计算 ======================
    def _fit_plane_and_calculate_equation(self, points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        min_idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, min_idx]
        if np.dot(normal, centroid - self.coordinate_system['origin']) < 0:
            normal = -normal
        A, B, C = normal
        D = -np.dot(normal, centroid)
        distances = np.abs(A*points[:,0] + B*points[:,1] + C*points[:,2] + D) / np.linalg.norm(normal)
        rmse = np.sqrt(np.mean(distances**2))
        bounds = self._calculate_surface_bounds(points, normal, centroid)

        # >>>>>>>  产状：走向 / 倾向 / 倾角  <<<<<<<
        nx, ny, nz = normal
        # 1. 倾角
        dip_deg = np.degrees(np.arccos(np.abs(nz)))                 # 0~90°
        # 2. 走向
        if np.isclose(nz, 0, atol=1e-8):                           # 垂直面
            strike_vec = np.array([ny, -nx, 0])
        else:
            strike_vec = np.array([ny, -nx, 0])                     # 水平投影
        strike_vec /= np.linalg.norm(strike_vec)
        strike = np.degrees(np.arctan2(strike_vec[1], strike_vec[0]))
        strike = (90 - strike) % 360                                # 北为0，顺时针
        # 3. 倾向 = 走向 + 90°
        dip_dir = (strike + 90) % 360
        attitude = f"走向 {strike:05.1f}° / 倾向 {dip_dir:05.1f}° / 倾角 {dip_deg:04.1f}°"

        return {'normal_vector': normal,
                'centroid': centroid,
                'equation': {'A': A, 'B': B, 'C': C, 'D': D,
                             'string': f"{A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0"},
                'rmse': rmse,
                'bounds': bounds,
                'attitude': attitude}          # ← 新增

    def _calculate_surface_bounds(self, points, normal, centroid):
        if np.abs(normal[0]) > np.abs(normal[1]):
            basis1 = np.array([-normal[2], 0, normal[0]])
        else:
            basis1 = np.array([0, normal[2], -normal[1]])
        basis1 /= np.linalg.norm(basis1)
        basis2 = np.cross(normal, basis1)
        proj = np.column_stack([np.dot(points - centroid, basis1),
                                np.dot(points - centroid, basis2)])
        hull = ConvexHull(proj)
        hull3d = centroid + proj[hull.vertices][:, 0:1]*basis1 + proj[hull.vertices][:, 1:2]*basis2
        return hull3d

    # ====================== 打印 ======================
    def print_surface_equations(self):
        print("\n" + "=" * 60)
        print("表面方程汇总:")
        print("=" * 60)
        for i, surface in enumerate(self.surfaces):
            print(f"\n表面 {i + 1}:")
            print(f"  点数: {surface['point_count']}")
            print(f"  法向量: [{surface['normal_vector'][0]:.4f}, "
                  f"{surface['normal_vector'][1]:.4f}, {surface['normal_vector'][2]:.4f}]")
            print(f"  质心: [{surface['centroid'][0]:.4f}, "
                  f"{surface['centroid'][1]:.4f}, {surface['centroid'][2]:.4f}]")
            print(f"  平面方程: {surface['equation']['string']}")
            print(f"  拟合误差 (RMSE): {surface['rmse']:.6f}")
            print(f"  产状: {surface['attitude']}")          # ← 新增

    # ------------------ 可视化（完全未改动） ------------------
    def visualize_in_3d_coordinate_system(self, point_cloud=None,
                                          show_original_points=False,
                                          show_points=False):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        origin = np.array([0., 0., 0.])
        if hasattr(self.coordinate_system['bounds'], 'max'):
            axis_length = np.max(self.coordinate_system['bounds']['max'] -
                                 self.coordinate_system['bounds']['min']) * 0.3
        else:
            all_pts = np.vstack([s['points'] for s in self.surfaces])
            axis_length = (all_pts.max(0) - all_pts.min(0)).max() * 0.3
        for axis, color, name in zip([self.coordinate_system['x_axis'],
                                      self.coordinate_system['y_axis'],
                                      self.coordinate_system['z_axis']],
                                     ['red', 'green', 'blue'],
                                     ['X', 'Y', 'Z']):
            ax.quiver(*origin, *(axis * axis_length), color=color, linewidth=2,
                      arrow_length_ratio=0.1, label=f'{name}轴')
        ax.scatter(*origin, color='black', s=50, marker='o', label='原点')
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, surface in enumerate(self.surfaces):
            color = colors[i % len(colors)]
            hull_pts = surface['bounds']
            if len(hull_pts) >= 3:
                hull2d = ConvexHull(hull_pts[:, :2])
                for simplex in hull2d.simplices:
                    tri = hull_pts[simplex]
                    poly = Poly3DCollection([tri], alpha=0.45, facecolor=color, edgecolor='k', linewidths=0.4)
                    ax.add_collection3d(poly)
            normal = surface['normal_vector']
            centroid = surface['centroid']
            ax.quiver(*centroid, *(normal * axis_length * 0.5), color=color,
                      linewidth=2, arrow_length_ratio=0.1,
                      label=f'表面 {i + 1} 法向量')
        all_vtx = np.vstack([s['bounds'] for s in self.surfaces])
        min_lim = all_vtx.min(0); max_lim = all_vtx.max(0); center_lim = (min_lim + max_lim) / 2
        max_range = (max_lim - min_lim).max() / 2
        ax.set_xlim(center_lim[0] - max_range, center_lim[0] + max_range)
        ax.set_ylim(center_lim[1] - max_range, center_lim[1] + max_range)
        ax.set_zlim(center_lim[2] - max_range, center_lim[2] + max_range)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('点云表面提取结果 - 所有平面同时显示')
        plt.tight_layout(); plt.show()

# ---------- 工具函数 ----------
def find_point_cloud_files():
    desktop = r"C:\Users\王言俊\Desktop"
    if not os.path.exists(desktop): return []
    return [os.path.join(desktop, f) for f in os.listdir(desktop) if f.lower().endswith('.txt')]

def load_point_cloud_file(file_path):
    print(f"正在加载点云文件: {file_path}")
    try:
        if not os.path.isfile(file_path): raise FileNotFoundError("文件不存在")
        raw = np.genfromtxt(file_path, delimiter=None, comments=None, encoding='utf-8')
        raw = raw[~np.isnan(raw).any(axis=1)]
        if raw.size == 0: raise ValueError("文件为空或没有有效数字！")
        if raw.ndim == 1: raw = raw.reshape(-1, 1)
        if raw.shape[1] < 3: raise ValueError(f"列数不足3，当前只有{raw.shape[1]}列")
        point_cloud = raw[:, :3]
        print("有效点云 shape:", point_cloud.shape)
        return point_cloud
    except Exception as e:
        print(f"加载点云文件时出错: {e}")
        return None

def load_point_cloud_from_e_drive():
    files = find_point_cloud_files()
    if not files:
        print("错误：桌面未找到任何TXT文件"); sys.exit(1)
    for i, p in enumerate(files, 1): print(f"  {i}. {os.path.basename(p)}")
    idx = 0 if len(files) == 1 else int(input("请选择文件编号：") or 1) - 1
    pc = load_point_cloud_file(files[idx])
    if pc is None: sys.exit(1)
    return pc, files[idx]

# ---------- 主程序 ----------
if __name__ == "__main__":
    surface_generator = PointCloudSurfaceGenerator()
    print("=" * 60)
    print("点云表面提取系统")
    print("=" * 60)

    point_cloud, file_path = load_point_cloud_from_e_drive()
    if point_cloud is None or point_cloud.size == 0:
        print("点云加载失败，程序终止。"); sys.exit(1)

    eps_value = 0.15; min_samples_value = 25
    print(f"\n当前参数: eps = {eps_value},  min_samples = {min_samples_value}")
    if input("是否调整参数？(y/n, 默认n): ").lower() == 'y':
        eps_value = float(input(f"新eps (当前{eps_value}): ") or eps_value)
        min_samples_value = int(input(f"新min_samples (当前{min_samples_value}): ") or min_samples_value)

    surfaces, labels = surface_generator.extract_surfaces(point_cloud, eps=eps_value, min_samples=min_samples_value)
    surface_generator.print_surface_equations()
    print("生成可视化结果...")
    surface_generator.visualize_in_3d_coordinate_system(point_cloud,
                                                        show_original_points=False,
                                                        show_points=False)

    # 保存结果（含产状）
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        result_file = fr"C:\Users\王言俊\Desktop\{base_name}_surface_equations.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("点云表面提取结果\n" + "=" * 50 + "\n")
            f.write(f"点云文件: {file_path}\n")
            f.write(f"点云大小: {point_cloud.shape}\n")
            f.write(f"参数: eps={eps_value}, min_samples={min_samples_value}\n\n")
            for i, s in enumerate(surfaces, 1):
                f.write(f"表面 {i}:\n")
                f.write(f"  点数: {s['point_count']}\n")
                f.write(f"  法向量: [{s['normal_vector'][0]:.4f}, "
                        f"{s['normal_vector'][1]:.4f}, {s['normal_vector'][2]:.4f}]\n")
                f.write(f"  质心: [{s['centroid'][0]:.4f}, "
                        f"{s['centroid'][1]:.4f}, {s['centroid'][2]:.4f}]\n")
                f.write(f"  平面方程: {s['equation']['string']}\n")
                f.write(f"  拟合误差 (RMSE): {s['rmse']:.6f}\n")
                f.write(f"  产状: {s['attitude']}\n\n")          # ← 新增
        print(f"结果已保存到 {result_file}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

    print("\n表面提取完成！")