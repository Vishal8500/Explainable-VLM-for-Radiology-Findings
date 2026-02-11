from datasets import load_dataset

ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")

# ds.save_to_disk(r"D:\CV PROJ\dataset")
print(ds)
print(ds["train"][0])