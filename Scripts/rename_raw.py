import os

src1 = r"D:\Data\cbct\image"
src2 = r"D:\Data\cbct\image4"

paths1 = os.listdir(src1)
paths2 = os.listdir(src2)


for p in paths2:
    suffix = p[-7:]
    print(suffix)
    for k in paths1:
        if suffix in k:
            old_path = os.path.join(src2, p)
            new_path = os.path.join(src2, k)

            if os.path.exists(new_path):
                print(f"Warning: File '{new_path}' already exists. Skipping rename.")
                continue

            print(f"Renaming '{old_path}' to '{new_path}'")
            os.rename(old_path, new_path)
