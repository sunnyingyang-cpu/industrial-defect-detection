import os
import yaml
import matplotlib.pyplot as plt

label_paths = ['../data/vehicle/labels/train', '../data/vehicle/labels/val']

names_fp = '../data/vehicle.yaml'
with open(names_fp, 'r', encoding='utf-8') as fp:
    names = yaml.load(fp, yaml.FullLoader)['names']

label_fp = []
for path in label_paths:
    label_fp += [os.path.join(path, fp) for fp in os.listdir(path)]

cls = {k: 0 for k in names.keys()}
for fp in label_fp:
    for b in open(fp, 'r').readlines():
        cls[eval(b.split()[0])] += 1
result = {names[k]: v for k, v in cls.items()}
print(result)

labels = list(result.keys())
values = list(result.values())

plt.bar(labels, values)
plt.ylabel('number')
plt.xlabel('category')
plt.show()
