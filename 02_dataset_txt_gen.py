import os


root_dir = "/test/data/hymenoptera_data/train"
target_dir = "ants"
img_path = os.listdir(os.path.join(root_dir, target_dir))
# label = target_dir.split('_')[0]
# print(label)
out_dir = "ants_label"
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(target_dir)
