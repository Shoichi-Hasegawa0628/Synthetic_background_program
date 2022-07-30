from torch.utils.data import Dataset


class YCBDatasetInfo(Dataset):
    raw_object_names = [
        "001_apple",
        "002_orange",
        "003_cracker",
        "004_chips_bag",
        "005_coffee",
        "006_muscat",
        "007_penguin_doll",
        "008_frog_shaped_sponge",
        "009_cup",
        "010_sponge"
    ]


    raw_object_id_to_name_dict = {
        1: "apple",
        2: "orange",
        3: "cracker",
        4: "chips_bag",
        5: "coffee",
        6: "muscat",
        7: "penguin_doll",
        8: "frog_shaped_sponge",
        9: "cup",
        10: "sponge"
    }

    object_id_to_index_dict = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9
    }

    # url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/objects.json"


if __name__ == '__main__':
    x = [f'"{v}"' for _, v in YCBDatasetInfo.raw_object_id_to_name_dict.items()]
    print(len(x))
    print("[", ", ".join(x), "]")
