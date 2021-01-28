import os
import math
import json
import copy
import operator
from pathlib import Path
from functools import reduce
from collections import Counter

import numpy as np
from imutils import paths
from shapely.geometry import Polygon

def most_common(a):
    count = {}
    max_count = 0
    result = a[0]
    for i in a:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
        if count[i] > max_count:
            max_count = count[i]
            result = i
    
    return result


def smoothen(seq, radius=2):
    if radius == 0:
        return seq
    
    result = [-1] * len(seq)
    result[:radius] = seq[:radius]
    result[-radius:] = seq[-radius:]
    
    for i in range(radius, len(seq) - radius):
        result[i] = most_common(seq[i - radius : i + radius + 1])
    
    for i in range(radius, len(result) - radius):
        result[i] = most_common(result[i - radius : i + radius + 1])
    
    return result

def convert_SO_to_BIOES(tags):
    # print(tags)
    new_tags = []
    
    i = 0
    while i < len(tags):
        if tags[i] == 'O':
            new_tags.append(tags[i])
            i += 1
        elif tags[i].startswith('S-'):
            if i + 1 == len(tags) or tags[i+1] != tags[i]:
                new_tags.append(tags[i])
                i += 1
            else:
                new_tags.append(tags[i].replace('S-', 'B-'))
                j = i + 1
                while j < len(tags) and tags[j] == tags[i]:
                    new_tags.append(tags[i].replace('S-', 'I-'))
                    j += 1
                new_tags[-1] = new_tags[-1].replace('I-', 'E-')
                i = j
            
        else:
            raise Exception('Invalid format!')
    
    return new_tags


def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def sort_funsd_reading_order(lines):
    """ Sort cell list to create the right reading order using their locations
    Parameters
    ----------
    lines: list of cells to sort
    Returns
    -------
    a list of cell lists in the right reading order that contain no key or start with a key and contain no other key
    """
    sorted_list = []
    
    if len(lines) == 0:
        return lines
    
    while len(lines) > 1:
        topleft_line = lines[0]
        for line in lines[1:]:
            topleft_line_pos = topleft_line['box']
            topleft_line_center_y = (topleft_line_pos[1] + 
                                     topleft_line_pos[3]) / 2
            x1, y1, x2, y2 = line['box']
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            cell_h = y2 - y1
            if box_center_y <= topleft_line_center_y - cell_h / 2:
                topleft_line = line
                continue
            if box_center_x < topleft_line_pos[2] and box_center_y < topleft_line_pos[3]:
                topleft_line = line
                continue
        sorted_list.append(topleft_line)
        lines.remove(topleft_line)
    
    sorted_list.append(lines[0])
    
    return sorted_list


def dist_points(p_a, p_b):
    return np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2)


def get_mid_points(p_a, p_b, ratio):
    return int(p_a[0] * ratio + p_b[0] * (1 - ratio)), int(p_a[1] * ratio + p_b[1] * (1 - ratio))


def get_closest_point(ref_point, point_lists):
    dist_list = [dist_points(ref_point, p) for p in point_lists]
    return point_lists[np.argmin(dist_list)]


def divide_polygon_by_ratio(points, offset_ratio, width_ratio):

    x1, y1, x2, y2 = box = convert_pts2box(points)
    bounding_pts = (x1, y1), (x2, y1), (x2, y2), (x1, y2)

    # print('points', points)
    points = [get_closest_point(bounding_pts[i], points) for i in range(4)]
    # print('after', points)

    # if dist_points(points[0], points[1]) > dist_points(points[0], points[3]):
    #     width_pairs = [(0, 1), (2, 3)]
    # else:
    #     width_pairs = [(0, 3), (2, 1)]
    #
    # # reverse order according to distance to top left corner
    # if dist_points(box[:2], points[width_pairs[0][1]]) < dist_points(box[:2], points[width_pairs[0][0]]):
    #     width_pairs[0] = width_pairs[0][::-1]
    #
    # # reverse order according to distance to bottom left corner
    # if dist_points((box[0], box[3]), points[width_pairs[1][1]]) < dist_points((box[0], box[3]), points[width_pairs[1][0]]):
    #     width_pairs[1] = width_pairs[1][::-1]


    top_points = [get_mid_points(points[0], points[1], offset_ratio), get_mid_points(
        points[0], points[1], offset_ratio - width_ratio)]

    bottom_points = [get_mid_points(points[3], points[2], offset_ratio), get_mid_points(
        points[3], points[2], offset_ratio - width_ratio)]


    # top_points = [get_mid_points(points[width_pairs[0][0]], points[width_pairs[0][1]], offset_ratio), get_mid_points(
    #     points[width_pairs[0][0]], points[width_pairs[0][1]], offset_ratio + width_ratio)]
    #
    # bottom_points = [get_mid_points(points[width_pairs[1][0]], points[width_pairs[1][1]], offset_ratio), get_mid_points(
    #     points[width_pairs[1][0]], points[width_pairs[1][1]], offset_ratio + width_ratio)]

    return top_points + bottom_points[::-1]


def convert_pts2box(pts):
    list_x = [p[0] for p in pts]
    list_y = [p[1] for p in pts]

    return int(min(list_x)), int(min(list_y)), int(max(list_x)), int(max(list_y))


def sorted_clockwise_order(points):
    center = tuple(map(operator.truediv, 
                       reduce(lambda x, y: map(operator.add, x, y), points), 
                       [len(points)] * 2))
    
    def __clockwise_order(point):
        diff_vec = tuple(map(operator.sub, point, center))[::-1]
        angle = math.degrees(math.atan2(*diff_vec)) % 360
        
        return -135 - angle
    
    return sorted(points, key=__clockwise_order)


def union_datapile_flow_output(datapile_data, flow_output_data, 
                               targeted_fields=None):
    '''
    output = text lines containing targeted fields from flow output 
           + text lines not containing targeted fields from datapile label
           
    output is in datapile format
    '''
    
    def __check_overlap(output_polygon, datapile_targeted_regions):
        p1 = Polygon(output_polygon)
        for r in datapile_targeted_regions:
            if r['shape_attributes']['name'] == 'rect':
                x = r['shape_attributes']['x']
                y = r['shape_attributes']['y']
                w = r['shape_attributes']['width']
                h = r['shape_attributes']['height']
                datapile_polygon = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
            elif r['shape_attributes']['name'] == 'polygon':
                datapile_polygon = [(x, y) for x, y 
                                    in zip(r['shape_attributes']['all_points_x'], 
                                           r['shape_attributes']['all_points_y'])
                                    ]
            datapile_polygon = sorted_clockwise_order(datapile_polygon)
            p2 = Polygon(datapile_polygon)
            
            if p2.area < 1:
                continue
            
            try:
                p1.intersection(p2).area / p2.area
            except:
                from IPython import embed
                embed()
             
            if p1.intersection(p2).area / p2.area > 0.2:
                return True
        
        return False 
    
    datapile_regions = datapile_data['attributes']['_via_img_metadata']['regions']
    
    targeted_regions = []
    non_targeted_regions = []
    
    for r in datapile_regions:
        formal_key = r['region_attributes']['formal_key']
        if (len(formal_key) > 0 
            and (targeted_fields is None or formal_key in targeted_fields)):
            targeted_regions.append(r)
    
    for output_region in flow_output_data:
        output_polygon = sorted_clockwise_order(output_region['location'])
        if not __check_overlap(output_polygon, targeted_regions):
            non_targeted_regions.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [p[0] for p in output_polygon],
                    "all_points_y": [p[1] for p in output_polygon],
                },
                "region_attributes": {
                    "formal_key": "",
                    "key_type": "other",
                    "label": output_region['text'],
                    "note": "",
                    "text_category": "",
                    "text_type": ""
                }
            })
    
    union_in_datapile_format = copy.deepcopy(datapile_data)
    union_in_datapile_format['attributes']['_via_img_metadata']['regions'] = \
        targeted_regions + non_targeted_regions
    
    return union_in_datapile_format

if __name__ == '__main__':
    datapile_dir = r'D:\Experiments\layout-lm\data_raw\sompo_mr\train'
    flow_output_dir = r'D:\Experiments\layout-lm\data_raw\sompo_mr\input_output\debugs'
    output_dir = Path(r'D:\Experiments\layout-lm\data_raw\sompo_mr') / 'union_with_flow_output/train'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_map = {}
    for p in paths.list_files(datapile_dir, validExts=('.json')):
        k = os.path.splitext(os.path.basename(p))[0]
        data_map[k] = {'datapile_label': p}
    
    for p in paths.list_files(flow_output_dir, validExts=('.json')):
        if os.path.basename(p) != 'ocr_output.json':
            continue
        p = os.path.normpath(p)
        k = p.split(os.sep)[-2]
        if k in data_map:
            data_map[k]['flow_output'] = p
    
    for k, v in data_map.items():
        if 'datapile_label' not in v or 'flow_output' not in v:
            continue
        
        print('-' * 50)
        print('label:', v['datapile_label'])
        print('flow output:', v['flow_output'])
        
        with open(v['datapile_label'], 'r', encoding='utf8') as f:
            datapile_data = json.load(f)
        with open(v['flow_output'], 'r', encoding='utf8') as f:
            flow_output_data = json.load(f)
    
        union_data = union_datapile_flow_output(datapile_data, flow_output_data)
        
        with (output_dir / (k + '.json')).open(mode='w', encoding='utf8') as f:
            json.dump(union_data, f, ensure_ascii=False, indent=4)
    