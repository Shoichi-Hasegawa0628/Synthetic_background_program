from torch.utils.data import Dataset


class YCBDatasetInfo(Dataset):
    raw_object_names = [
        "001_apple",
        "002_cracker",
        "003_coffee",
        "004_penguin_doll",
        "005_frog_shaped_sponge",
        "006_cup"
    ]


    raw_object_id_to_name_dict = {
        1: "apple",
        2: "cracker",
        3: "coffee",
        4: "penguin_doll",
        5: "frog_shaped_sponge",
        6: "cup"
    }

    object_id_to_index_dict = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5
    }

    # url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/objects.json"


if __name__ == '__main__':
    x = [f'"{v}"' for _, v in YCBDatasetInfo.raw_object_id_to_name_dict.items()]
    print(len(x))
    print("[", ", ".join(x), "]")
