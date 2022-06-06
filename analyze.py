import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme()

results_acc = {'gaussian noise': np.zeros(6), 'gaussian blur': np.zeros(6), 'contrast transform': np.zeros(6), 'jpeg transform': np.zeros(6), 'rotation transform': np.zeros(6)}
results_steps = {'gaussian noise': np.zeros(6), 'gaussian blur': np.zeros(6), 'contrast transform': np.zeros(6), 'jpeg transform': np.zeros(6), 'rotation transform': np.zeros(6)}

f = open('transforms.txt')
transforms = {}
for line in f:
    a = line.split()
    transforms[int(a[0])] = [a[1], ' '.join(a[2:])]

directory='./results'
count = 0
for filename in os.listdir(directory):
    count += 1
    file = os.path.join(directory, filename)
    accuracies = []
    steps = []


    with open(file) as f:
        contents = f.readlines()

        use = 0
        idx_s, idx_a = [], []
        for line in contents:
                
            # save accuracies and steps
            if '/accuracy/dataloader_idx' in line:
                idx = int(line.split('test_')[1].split('/')[0].strip())
                acc = float(line.split(': ')[1].split(',')[0].strip())
                if idx in idx_a:
                    continue
                idx_a.append(idx)

                if idx == 0:
                    for key in results_acc.keys():
                        results_acc[key][0] += acc
                else:
                    s = int(transforms[idx][0])
                    t = transforms[idx][1]
                    results_acc[t][s] += acc
            elif '/steps/dataloader_idx' in line:
                idx = int(line.split('test_')[1].split('/')[0].strip())
                steps = float(line.split(': ')[1].split('}')[0].split(',')[0].strip())
                if idx in idx_s:
                    continue
                idx_s.append(idx)
                if idx == 0:
                    idx0 = steps
                    for key in results_steps.keys():
                        results_steps[key][0] += 100
                else:
                    s = int(transforms[idx][0])
                    t = transforms[idx][1]
                    results_steps[t][s] += (steps/idx0)*100

for key in results_acc.keys():
    plt.plot(range(6), results_acc[key]/count, label=key)

plt.legend()
plt.xlabel('severity')
plt.ylabel('accuracy')
plt.ylabel('average accuracy')
plt.savefig('accuracy.png')    
plt.clf()


for key in results_steps.keys():
    plt.plot(range(6), results_steps[key]/count, label=key)

plt.legend()
plt.xlabel('severity')
plt.ylabel('%')
plt.title('average relative growth w.r.t. severity 0')

plt.savefig('steps.png')    