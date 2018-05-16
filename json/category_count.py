# coding=utf-8
import json
from collections import defaultdict
import argparse
import os

HEAD = "id {:<2} total {:<2} difficult {:<2} category {:<18}\n".format("", "", "", "")
FORMAT = "{:<6} {:<9} {:<11} {:<8}\n"

category_id = {
    0: "空气内循环模式",
    1: "ECO启动/停止功能",
    2: "智能气候控制同步",
    3: "智能泊车",
    4: "电控车辆稳定行驶系统",
    5: "遮阳帘",
    6: "360度摄像头",
    7: "动态操控选择控制器",
    8: "SUV空气悬挂",
    9: "下坡车速控制系统",
    10: "主动式车道保持辅助系统",
    11: "智能限距功能启用",
    12: "安全带递送器",
    13: "智能限距功能启用"
}

parse = argparse.ArgumentParser()
parse.add_argument("-dir", dest="dir",
                   default="/Users/wangbin/PycharmProjects/IBM/data/", help="json file directory")
args = parse.parse_args()


def count_objects_per_file(filenames, output):
    output = output + "category_count.txt"
    total = defaultdict(int)
    difficult = defaultdict(int)

    for filename in filenames:
        with open(filename) as fr:
            annotation = json.load(fr)
            for key in annotation:
                num_object_per_image = len(annotation[key]["regions"])
                if num_object_per_image:
                    for region in annotation[key]["regions"].values():
                        category = region["region_attributes"]["object"].split("_")
                        if len(category[0]) < 3:
                            total[category[0]] += 1
                        if len(category) == 2:
                            difficult[category[0]] += 1
    with open(output, "w+") as fw:
        fw.write(HEAD)
        print(HEAD, end="")
        for key in sorted(total, key=lambda k: int(k)):
            stat = FORMAT.format(key, total[key], difficult[key], category_id[int(key)])
            fw.write(stat)
            print(stat, end="")


if __name__ == "__main__":

    json_files = []
    for json_file in os.listdir(args.dir):
        abspath = "".join([args.dir, json_file])
        if os.path.isfile(abspath) and json_file.endswith(".json"):
            json_files.append(abspath)
    count_objects_per_file(json_files, args.dir)
