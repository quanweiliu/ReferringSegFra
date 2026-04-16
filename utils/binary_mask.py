from itertools import count

import cv2
import numpy as np
import os
import json
import rasterio
from tqdm import tqdm


# 定义类别和对应的 BGR 颜色 
color_map = {
    "Impervious_surface": [255, 255, 255],      #
    "Building": [255, 0, 0],                    #
    "Car": [0, 255, 255],                       #
    "Tree": [0, 255, 0],                        #
    "LowVeg": [255, 255, 0]                     #
}


suffix = {
    "Impervious_surface": "0",
    "Building": "1",
    "Car": "2",
    "Tree": "3",
    "LowVeg": "4"
}


def get_color_stats(img):
    pixels = img.reshape(-1, img.shape[-1])
    
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    return unique_colors, counts


def extract_binary_masks(mask_dir, output_root):

    root_save_path = os.path.join(output_root, 'binary_masks')
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    
    # 为每个类别创建子文件夹
    for cls_name in color_map.keys():
        os.makedirs(os.path.join(root_save_path, cls_name), exist_ok=True)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    # print(f"Found {len(mask_files)} mask files to process.")
    
    metadata = {} # 用于存储每个 Patch 包含哪些类，为下一步生成 Text 做准备

    count_index = 0
    for filename in tqdm(mask_files):
        print("Processing", filename)
        mask_path = os.path.join(mask_dir, filename)
        # print("output filename", output_filename)
        # img = cv2.imread(mask_path)
        img = rasterio.open(mask_path).read().transpose(1, 2, 0)
        img_h, img_w, _ = img.shape

        # # 辅助统计信息
        # colors, counts = get_color_stats(img)
        # for c, n in zip(colors, counts):
        #     print("color",  c, "counts", n)
        # break
        
        present_classes = []
        
        for index, (cls_name, color) in enumerate(color_map.items()):
            output_filename = os.path.splitext(filename)[0] + '_' + str(index) + '.tif'
            
            # 创建二值掩码：颜色匹配的地方为 255，其余为 0
            binary_mask = cv2.inRange(img, np.array(color), np.array(color))
            
            # 统计像素点，如果该类像素占比超过 0.1%，则保存并记录
            if np.count_nonzero(binary_mask) > (img_h * img_w * 0.001):
                count_index += 1
                sub_save_path = os.path.join(root_save_path, cls_name, output_filename)
                cv2.imwrite(sub_save_path, binary_mask)
                present_classes.append(cls_name)
        
        metadata[filename] = present_classes
    print(f"Total binary masks saved: {count_index}")

    # 将元数据保存为 JSON，方便下一步调用 Gemini API 生成文本
    with open(os.path.join(output_root, 'patch_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def enrich_metadata(binary_mask_root, output_root):
    # 读取原始基础元数据
    with open(os.path.join(output_root, 'patch_metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    enriched_data = {}
    
    # 噪声过滤阈值：过滤掉小于 20 像素的零散点 (约 0.15 平方米)
    # 对于 Car 这种小目标，如果面积太小可能只是分割残留
    min_area_threshold = 20 

    for patch_name, classes in tqdm(metadata.items()):
        patch_info = {"classes": {}}
        
        for cls in classes:
            update_patch_name = patch_name.split('.')[0] + '_' + suffix[cls] + '.tif'
            mask_path = os.path.join(binary_mask_root, cls, update_patch_name)

            if not os.path.exists(mask_path):
                continue

            # mask = rasterio.open(mask_path).read().transpose(1, 2, 0)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or np.count_nonzero(mask) == 0:
                continue

            h_img, w_img = mask.shape[:2]
            # assert h_img == 480 and w_img == 480, f"Unexpected image size: {mask_path} has shape {mask.shape}"
            
            # --- 1. 连接组件分析 (Connected Components) ---
            # 使用 8 连通域，stats 包含 [left, top, width, height, area]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            valid_instances = []
            for i in range(1, num_labels): # 跳过背景 label 0
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area_threshold:
                    valid_instances.append({
                        "area": area,
                        "bbox": stats[i, :4].tolist(),
                        "centroid": centroids[i].tolist()
                    })
            
            instance_count = len(valid_instances)
            if instance_count == 0:
                continue # 如果全是噪声，忽略该类别

            # --- 2. 基础全局特征 ---
            area_pixels = np.count_nonzero(mask)
            area_ratio = area_pixels / mask.size
            
            # 全局重心 (Moments)
            M = cv2.moments(mask)
            cX = (M["m10"]/M["m00"])/w_img if M["m00"] != 0 else 0.5
            cY = (M["m01"]/M["m00"])/h_img if M["m00"] != 0 else 0.5

            # 全局 BBox (用于判定跨度 is_spanning)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # [xmin, ymin, xmax, ymax] 归一化后存入 JSON
            global_bbox = [cmin/w_img, rmin/h_img, (cmax+1)/w_img, (rmax+1)/h_img]

            # --- 3. 几何特征提取 (基于最大主实例) ---
            # 找到面积最大的实例
            # main_inst = max(valid_instances, key=lambda x: x["area"])
            
            # 为了获取 solidity 和 aspect_ratio，我们需要最大实例的轮廓
            # 创建一个只包含最大实例的临时 mask
            # 找到 labels 中等于最大实例索引的区域 (索引为 stats 中的顺序，需对应)
            # 这里简单起见，直接从全局 mask 提取所有轮廓，选面积最大的
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main_cnt = max(contours, key=cv2.contourArea)

            # 3.1 旋转矩形长宽比 (Aspect Ratio)
            rect = cv2.minAreaRect(main_cnt)
            (_, _), (w, h), _ = rect
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            
            # 3.2 实心度 (Solidity)
            hull = cv2.convexHull(main_cnt)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(main_cnt) / hull_area if hull_area > 0 else 1.0
            
            # 3.3 延伸度 (Extent)
            x_b, y_b, w_b, h_b = cv2.boundingRect(main_cnt)
            extent = cv2.contourArea(main_cnt) / (w_b * h_b) if (w_b * h_b) > 0 else 0

            # --- 4. 存入 JSON 结构 ---
            patch_info["classes"][cls] = {
                "area_ratio": round(area_ratio, 3),        # 全局属性: 面积占比
                "instance_count": instance_count,          # 全局属性: 实例数量
                "position": [round(cX, 2), round(cY, 2)],  # 全局属性: 所有像素点的几何中心。
                "shape": {
                    "aspect_ratio": round(aspect_ratio, 2),  # 最大实例属性: 长宽比
                    "solidity": round(solidity, 2),          # 最大实例属性: 紧凑度
                    "extent": round(extent, 2)               # 最大实例属性: 延伸度
                },
                "bbox": [round(b, 3) for b in global_bbox] # 全局属性: 所有像素点的最小水平矩形。
            }

        if patch_info["classes"]:
            enriched_data[patch_name] = patch_info

    # 保存路径保持不变
    save_path = os.path.join(output_root, 'enriched_metadata.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=4)
    print(f"成功生成增强型元数据 {len(enriched_data)}，包含实例统计：{save_path}")


# --- 1. 语义提升逻辑 ---
# 这里的名字会影响其它属性的判定，所以谨慎修改，
# 修改需要检查有没有影响到后续的方位，规模和上下文判定逻辑
def get_natural_name(cls_name, m, all_m):
    s = m.get("shape", {})
    r = m.get("area_ratio", 0)         # 面积占比
    asp = s.get("aspect_ratio", 1.0)   # 长宽比
    sol = s.get("solidity", 1.0)       # 紧凑度
    count = m.get("instance_count", 1)
    # 判定是否触碰边缘
    b = m.get("bbox", [0, 0, 1, 1])
    # 统一边缘判定阈值，并且只要触碰任一边就认为是被截断的
    is_trunc = b[0] <= 0.01 or b[1] <= 0.01 or b[2] >= 0.99 or b[3] >= 0.99


    # --- 1. 不透水面 (Impervious Surface) ---
    if cls_name == "Impervious_surface":
        has_cars = "Car" in all_m

        # --- 逻辑 1：细长形状 (Street 优先) ---
        if asp > 2.2 or (asp > 1.8 and is_trunc):
            # asp 每增加 1 奖励 0.05，最高到 0.95
            conf = min(0.85 + (asp - 2.2) * 0.03, 0.98)
            if has_cars:
                # 这种描述在 RRSIS 中非常受欢迎，因为它同时描述了两个类别
                return "street with parked vehicles", conf
            return "street", conf

        # --- 逻辑 2：块状形状 (可能是停车场或硬地) ---
        if r > 0.2 and r < 0.5 and sol < 0.7:
            return "road intersection", 0.90

        # if r >= 0.5:
        #     return "paved plaza", 0.90
                    
        # 兜底：铺装地面
        return "paved surface", 0.85

    # --- 2. 建筑 (Vaihingen 核心地物) ---
    if cls_name == "Building":
        if r > 0.40 and sol > 0.85:
            conf = 0.95 if sol > 0.92 else 0.90
            if is_trunc:
                return "section of a large building", conf
            return "large building", conf
        elif count > 1 and sol < 0.7: # 多个建筑且分布不紧凑
            return "cluster of buildings", 0.85
        else:
            return "detached house", 0.95

    if cls_name == "LowVeg":
        if "Building" in all_m:
            return "lawn", 0.90
        return "grassy field", 0.95

    if cls_name == "Tree":
        if r > 0.12:
            return "wooded patch", 0.90
        return "trees", 0.99
    
    if cls_name == "Car": 
        if count >= 10:
            return "cars in the parking lot", 0.95
        if count >= 8:
            return "cars in the roadside parking", 0.85
        return "vehicle", 0.95
    
    return cls_name.lower(), 0.5


# --- 2. 九宫格方位判定 (真正启用映射逻辑) ---
def get_spatial_description(b, e, target_name, ratio, count):
    x_min, y_min, x_max, y_max = b
    
    # 定义映射
    h_mapping = {0: "left", 1: "central", 2: "right"}
    v_mapping = {0: "top", 1: "middle", 2: "bottom"}
    
    # 判定占据了哪些列和行
    occ_h = [i for i in range(3) if x_min < (i+1)*0.334 and x_max > i*0.333]
    occ_v = [i for i in range(3) if y_min < (i+1)*0.334 and y_max > i*0.333]

    v_labels = [v_mapping[v] for v in occ_v]
    h_labels = [h_mapping[h] for h in occ_h]
    
    # 如果是多个离散物体散布
    if count > 2 and ("street" not in target_name \
                      and "road" not in target_name\
                      and "paved" not in target_name) and ratio < 0.3:
        # 获取垂直方向描述 (top, middle, bottom)
        v_desc = " and ".join(v_labels)
        v_s = "zones" if len(v_labels) > 1 else "zone"
        
        # 获取水平方向描述 (left, central, right)
        h_desc = " and ".join(h_labels)
        h_s = "sectors" if len(h_labels) > 1 else "sector"
        
        # 组合成一个完整的分布描述
        return f"scattered across the {v_desc} {v_s} and {h_desc} {h_s}"

    # 针对车辆等离散目标的特殊描述
    if count >= 2 and ("vehicle" in target_name \
                       or "Vehicles" in target_name \
                       or "cars" in target_name \
                        or "house" in target_name ):
        # 获取垂直方向描述 (top, middle, bottom)
        v_desc = " and ".join(v_labels)
        v_s = "zones" if len(v_labels) > 1 else "zone"
        
        # 获取水平方向描述 (left, central, right)
        h_desc = " and ".join(h_labels)
        h_s = "sectors" if len(h_labels) > 1 else "sector"
        
        # 组合成一个完整的分布描述
        return f"scattered across the {v_desc} {v_s} and {h_desc} {h_s}"

    # 3.1 跨度判定 (Spanning/Stretching)
    is_spanning_h = len(occ_h) == 3 or (x_min <= 0.05 and x_max >= 0.95)
    is_spanning_v = len(occ_v) == 3 or (y_min <= 0.05 and y_max >= 0.95)

    loc_parts = []
    # --- 情况 A: 双向贯穿 (对角线) ---
    if is_spanning_h and is_spanning_v:
        if ratio > 0.35:
            loc_parts.append("occupying a major portion of the patch")
        else:
            # 精准判定四个角点的触碰状态
            tl = "top" in e and "left" in e
            tr = "top" in e and "right" in e
            bl = "bottom" in e and "left" in e
            br = "bottom" in e and "right" in e
            
            if tl and br:
                loc_parts.append("diagonally traversing from the top-left towards the bottom-right")
            elif bl and tr:
                loc_parts.append("diagonally traversing from the bottom-left towards the top-right")
            else:
                # 如果不是标准的角对角，但确实横竖都贯穿了
                loc_parts.append("diagonally traversing across the patch")

    # --- 情况 B: 仅横向贯穿 ---
    elif is_spanning_h:
        v_idx = int(round(np.mean(occ_v))) if occ_v else 1
        v_pos = v_mapping[v_idx]
        # 针对小比例的横向地物
        if ratio < 0.1:
            loc_parts.append(f"passing through the {v_pos} edge of the patch")
        else:
            verb = "stretching" if "street" in target_name else "extending"
            loc_parts.append(f"{verb} horizontally across the {v_pos} of the patch")

    # --- 情况 C: 仅纵向贯穿 ---
    elif is_spanning_v:
        h_idx = int(round(np.mean(occ_h))) if occ_h else 1
        h_pos = h_mapping[h_idx]
        # 针对小比例的横向地物（如你图里的情况）
        if ratio < 0.1:
            loc_parts.append(f"passing through the {h_pos} edge of the patch")
        else:
            verb = "stretching" if "street" in target_name else "extending"
            loc_parts.append(f"{verb} vertically across the {h_pos} of the patch")
    
    # --- 情况 D: 常规方位 (九宫格) ---
    else:
        v_labels = [v_mapping[v] for v in occ_v]
        h_labels = [h_mapping[h] for h in occ_h]
        
        if len(v_labels) == 1 and len(h_labels) == 1:
            # 典型的单一方位
            loc_parts.append(f"in the {v_labels[0]}-{h_labels[0]} area")
        else:
            # 组合方位，例如 "spanning the top and middle-central areas"
            v_s = "zones" if len(v_labels) > 1 else "zone"
            h_s = "sectors" if len(h_labels) > 1 else "sector"

            v_desc = " and ".join(v_labels)
            h_desc = " and ".join(h_labels)
            loc_parts.append(f"occupying the {v_desc} {v_s} and {h_desc} {h_s}")

    # 3.3 补充边缘信息 (防止和前面的描述冲突)
    # 如果前面已经说了 stretching across patch，通常不需要再说 extending to edge
    if e and not any(kw in loc_parts[0] for kw in ["patch", "portion", "traversing"]):
        loc_parts.append(f"extending to the {', '.join(e)} edge")

    return " ".join(loc_parts)


# --- 3. 规模判定 (基于 0.4 的 Extensive 阈值) ---
def get_refined_scale(natural_name, ratio, count):
    # 针对连续地物（路、铺装面）
    if "street" in natural_name \
        or "paved" in natural_name \
        or "road" in natural_name:
        if ratio < 0.05: return "narrow" # 比如你图中小角的那一点路
        if ratio < 0.15: return "compact"
        if ratio < 0.3:  return "substantial"
        return "extensive"
    
    # 针对离散目标（车、房）优先使用数量描述
    if any(word in natural_name for word in ["cars", "vehicle", \
                                             "house", "building", "buildings"]):
        if count == 1: return "a single"
        if count <= 4: return "a small group of"
        if count <= 8: return "a cluster of"
        return "a large number of"

    # 默认（针对建筑、植被）
    if ratio < 0.1:  return "minor"
    if ratio < 0.25: return "moderate"
    if ratio < 0.45: return "large-scale"
    return "dominant"


# --- 6. 相对关系 (保持原有逻辑) ---
def get_contextual_relation(target_pos, target_bbox, other_data, other_name):
    """
    计算目标地物与参照地物之间的多级空间关系。
    
    参数:
    - target_pos: 目标质心 [x, y]
    - target_bbox: 目标 BBox [xmin, ymin, xmax, ymax]
    - other_data: 参照地物的原始元数据字典 (需含 position, bbox, instance_count)
    - other_name: 参照地物的自然语言名称 (已处理好单复数)
    
    返回:
    - 一段描述性的字符串，例如 "alongside the street"
    """
    # 1. 提取参照物数据
    o_pos = other_data.get("position", [0.5, 0.5])
    o_bbox = other_data.get("bbox", [0.0, 0.0, 1.0, 1.0])
    o_xmin, o_ymin, o_xmax, o_ymax = o_bbox
    t_xmin, t_ymin, t_xmax, t_ymax = target_bbox
    
    # 2. 计算基础距离
    dx, dy = target_pos[0] - o_pos[0], target_pos[1] - o_pos[1]
    dist_euclidean = np.sqrt(dx**2 + dy**2)
    
    # 3. 计算 BBox 最小间隙 (拓扑距离计算)
    # 计算水平和垂直方向的最短距离
    h_gap = max(0, max(t_xmin, o_xmin) - min(t_xmax, o_xmax))
    v_gap = max(0, max(t_ymin, o_ymin) - min(t_ymax, o_ymax))
    min_gap = np.sqrt(h_gap**2 + v_gap**2)
    # print(f"DEBUG: min_gap={min_gap:.4f}, dist={dist_euclidean:.4f}")

    # 4. 空间关系逻辑判定
    # --- 层级 A: 拓扑极近 (距离 < 0.03) ---
    if min_gap < 0.015:
        if "street" in other_name \
            or "road" in other_name: 
            rel_str = "alongside"
        elif "house" in other_name \
            or "building" in other_name:
            rel_str = "bordering"
        elif "plaza" in other_name:
            rel_str = "in"
        else:
            rel_str = "adjacent to"
            
    # --- 层级 B: 中近远距离 (8 方位描述) ---
    else:
        abs_dx, abs_dy = abs(dx), abs(dy)
        # 判定是否在对角线方向 (角度在 26.5° 到 63.5° 之间)
        if 0.5 < abs_dx / (abs_dy + 1e-5) < 2.0:
            ns = "north" if dy < 0 else "south"
            ew = "west" if dx < 0 else "east"
            rel_str = f"{ns}-{ew} of"
        else:
            # 主轴方位
            if abs_dy > abs_dx:
                rel_str = "north of" if dy < 0 else "south of"
            else:
                rel_str = "west of" if dx < 0 else "east of"

        # 如果距离比较近 (0.03 ~ 0.25)，我们引入 near/close to 修饰方位
        if 0.015 <= dist_euclidean < 0.15:
            return f"near and {rel_str} the {other_name}"
        elif 0.15 <= dist_euclidean < 0.25:
            return f"close to and {rel_str} the {other_name}"
    
    return f"{rel_str} the {other_name}"


def map_attributes(patch_id, target_class, class_data, all_classes_metrics):
    attr = {}
    shape = class_data.get("shape", {})
    pos = class_data.get("position", [0.5, 0.5])
    ratio = class_data.get("area_ratio", 0)
    bbox = class_data.get("bbox", [0.0, 0.0, 1.0, 1.0]) # [x_min, y_min, x_max, y_max]
    count = class_data.get("instance_count", 1) # 新增获取数量

    # --- 1. 物理边缘探测 ---
    edges = []
    if bbox[1] <= 0.01: edges.append("top")
    if bbox[3] >= 0.99: edges.append("bottom")
    if bbox[0] <= 0.01: edges.append("left")
    if bbox[2] >= 0.99: edges.append("right")
    attr["is_truncated"] = len(edges) > 0

    # ---  2. 获取自然名称 --- 
    natural_name, conf = get_natural_name(target_class, class_data, all_classes_metrics)
    attr["natural_name"] = f"partially shown {natural_name}" if (attr["is_truncated"] and "section" not in natural_name) else natural_name
    attr["confidence"] = round(conf, 2)

    # --- 3. 执行方位描述 ---
    attr["abs_location"] = get_spatial_description(bbox, edges, attr["natural_name"], ratio, count)
    
    # --- 4. 规模判定  ---
    attr["scale"] = get_refined_scale(natural_name, ratio, count)

    # --- 5. 几何特征 ---
    sol = shape.get("solidity", 1.0)
    ext = shape.get("extent", 1.0)
    asp = shape.get("aspect_ratio", 1.0)
    shape_list = []
    if count == 1: # 只有单体时几何形状才有意义
        if asp > 3.0: 
            shape_list.append("elongated")
        if sol > 0.85 and ext > 0.7: 
            shape_list.append("regular")
        elif sol < 0.6: 
            shape_list.append("irregularly shaped")
    attr["geometry"] = " and ".join(shape_list) if shape_list else "typical"  # 常规的

    # --- 6. 相对关系 (保持原有逻辑) ---
    enhanced_context = []
    # 按照面积排序，优先选大物体（如建筑、道路）作为参照物
    sorted_others = sorted(
        [(k, v) for k, v in all_classes_metrics.items() if k != target_class and k.lower() != "clutter"],
        key=lambda x: x[1].get("area_ratio", 0), reverse=True
    )

    for other_cls, other_data in sorted_others[:2]:
        
        # 获取参照物的自然语言名称 (它会自动处理 instance_count 带来的单复数)
        ctx_name, _ = get_natural_name(other_cls, other_data, all_classes_metrics)
        
        # 调用新函数获取关系
        relation_desc = get_contextual_relation(
            target_pos=pos, 
            target_bbox=bbox, 
            other_data=other_data, 
            other_name=ctx_name
        )
        
        enhanced_context.append(relation_desc)

    attr["context_relations"] = enhanced_context
    return attr


def prepare_gemini_inputs(input_json_path, output_root):
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    final_prompts = []
    for patch_id, content in data.items():
        all_metrics = content["classes"] # 这是一个包含该 Patch 所有类数据的字典
        
        for target_cls, metrics in all_metrics.items():
            instance_id = patch_id.split('.')[0] + '_' + suffix[target_cls] + '.tif'
            # print(f"target class: {target_cls}, metrics: {metrics.keys()}")
            instance_count = metrics.get("instance_count", 1)
            if target_cls.lower() == "clutter": continue
            
            # 这里的第四个参数传入了全量的 all_metrics
            attributes = map_attributes(patch_id, target_cls, metrics, all_metrics)
            
            final_prompts.append({
                "patch_id": patch_id,
                'instance_id': instance_id,
                "target_class": target_cls,
                "instance_count": instance_count,
                "attributes": attributes
            })
            
    with open(os.path.join(output_root, 'gemini_batch_input.json'), 'w') as f:
        json.dump(final_prompts, f, indent=4)
    print(f"已生成 {len(final_prompts)} 条待润色数据。")


# 提取二值 Mask
extract_binary_masks('/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/masks', \
                     '/home/icclab/Documents/lqw/DatasetMMF/VaihingenRef')


# # 加厚加厚的脚本
# enrich_metadata('/home/icclab/Documents/lqw/DatasetMMF/temp/binary_masks', \
#                    '/home/icclab/Documents/lqw/DatasetMMF/temp')


# # 运行属性映射与预处理脚本
# prepare_gemini_inputs('/home/icclab/Documents/lqw/DatasetMMF/temp/enriched_metadata.json', \
#                    '/home/icclab/Documents/lqw/DatasetMMF/temp')


# 特殊关注的 mask： val_95 / val_212  / val_364