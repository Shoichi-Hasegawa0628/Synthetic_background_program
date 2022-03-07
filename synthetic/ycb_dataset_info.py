from torch.utils.data import Dataset


class YCBDatasetInfo(Dataset):
    # raw_object_names = [
    #     "001_plate",
    #     "002_bowl",
    #     "003_pitcher_base",
    #     "004_banana",
    #     "005_apple",
    #     "006_orange",
    #     "007_cracker_box",
    #     "008_pudding_box",
    #     "009_chips_bag",
    #     "010_coffee",
    #     "011_muscat",
    #     "012_fruits_juice",
    #     "013_pig_doll",
    #     "014_sheep_doll",
    #     "015_penguin_doll",
    #     "016_airplane_toy",
    #     "017_car_toy",
    #     "018_truck_toy",
    #     "019_tooth_paste",
    #     "020_towel",
    #     "021_cup",
    #     "022_treatments",
    #     "023_sponge",
    #     "024_bath_slipper"
    # ]

    raw_object_names = [
        "001_banana",
        "002_apple",
        "003_orange",
        "004_cracker_box",
        "005_pudding_box",
        "006_pig_doll",
        "007_sheep_doll",
        "008_penguin_doll",
        "009_cup"
    ]

    # raw_object_names = [
    #     "001_toiletries",
    #     "002_snack",
    #     "003_doll",
    #     "004_fruits"
    # ]

    raw_object_id_to_name_dict = {
        # 1: "plate",
        # 2: "bowl",
        # 3: "pitcher_base",
        # 4: "banana",
        # 5: "apple",
        # 6: "orange",
        # 7: "cracker_box",
        # 8: "pudding_box",
        # 9: "chips_bag",
        # 10: "coffee",
        # 11: "muscat",
        # 12: "fruits_juice",
        # 13: "pig_doll",
        # 14: "sheep_doll",
        # 15: "penguin_doll",
        # 16: "airplane_toy",
        # 17: "car_toy",
        # 18: "truck_toy",
        # 19: "tooth_paste",
        # 20: "towel",
        # 21: "cup",
        # 22: "treatments",
        # 23: "sponge",
        # 24: "bath_slipper"

        1: "banana",
        2: "apple",
        3: "orange",
        4: "cracker_box",
        5: "pudding_box",
        6: "pig_doll",
        7: "sheep_doll",
        8: "penguin_doll",
        9: "cup"

        # 1: "toiletries",
        # 2: "snack",
        # 3: "doll",
        # 4: "fruits",

    }

    object_id_to_index_dict = {
        # 1: 0,
        # 2: 1,
        # 3: 2,
        # 4: 3,
        # 5: 4,
        # 6: 5,
        # 7: 6,
        # 8: 7,
        # 9: 8,
        # 10: 9,
        # 11: 10,
        # 12: 11,
        # 13: 12,
        # 14: 13,
        # 15: 14,
        # 16: 15,
        # 17: 16,
        # 18: 17,
        # 19: 18,
        # 20: 19,
        # 21: 20,
        # 22: 21,
        # 23: 22,
        # 24: 23

        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8

        # 1: 0,
        # 2: 1,
        # 3: 2,
        # 4: 3
    }

    # url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/objects.json"


if __name__ == '__main__':
    x = [f'"{v}"' for _, v in YCBDatasetInfo.raw_object_id_to_name_dict.items()]
    print(len(x))
    print("[", ", ".join(x), "]")
