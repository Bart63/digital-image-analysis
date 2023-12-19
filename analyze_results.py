# %%
import os
import json
import glob
import matplotlib.pyplot as plt

# %%
results_paths = glob.glob('./results/*.json')

# %%
def get_acc(results, top_n=5):
    successes = 0
    for query_fname, responses in results.items():
        query_lab = responses[0][1]

        responses.sort(key=lambda x: x[-1], reverse=True)
        succ = False
        for res in responses[:top_n]:
            if res[2] == query_lab:
                succ = True
        if succ:
            successes += 1
    return successes/len(results)

# %% Calc accuracy of all results
for result_path in results_paths:
    results = json.load(open(result_path))
    print(os.path.basename(result_path), round(get_acc(results, 1)*100, 2))
# %%
mn = 'ssim'
mns = ['mse', 'ssim', 'psnr']

fig, axs = plt.subplots(1,3)
fig.set_figwidth(19)
for ax, mn in zip(axs, mns):

    if mn == 'mse':
        ax.set_ylabel('Accuracy (%)')
    elif mn == 'ssim':
        ax.set_xlabel('Top N images')

    results_filenames = [
        f'{mn}_gc', f'{mn}_rgb', f'{mn}_hsv'
    ]

    res_dict = {k:[] for k in results_filenames}
    for fn in results_filenames:
        results = json.load(open(f'results/{fn}.json'))
        for top_n in [1, 5, 10]:
            acc = round(get_acc(results, top_n)*100, 2)
            res_dict[fn].append(acc)

    ax.grid(True)
    ax.plot(res_dict[f'{mn}_gc'], label=f'{mn}_gc', marker='o', linewidth=4, markersize=8)
    
    if mn == 'psnr':
        ax.plot(res_dict[f'{mn}_rgb'], label=f'{mn}_rgb', marker='o', linestyle='-.', linewidth=4, markersize=8)
    else:
        ax.plot(res_dict[f'{mn}_rgb'], label=f'{mn}_rgb', marker='o', linestyle='-', linewidth=4, markersize=8)
    ax.plot(res_dict[f'{mn}_hsv'], label=f'{mn}_hsv', marker='o', linewidth=4, markersize=8)
    ax.legend(loc='upper left')
    ax.set_ylim([0, 53])
    ax.set_xticks([0,1,2], ['1', '5', '10'])
axs[1].set_title(f'Accuracy of metric methods using different colorspaces')
# plt.xlabel('Top N images')
# plt.ylabel('Accuracy (%)')
plt.show()
# %%
res_dict

# %%

results_filenames = [
    f'statistical', f'haralick', f'hog'
]

res_dict = {k:[] for k in results_filenames}
for fn in results_filenames:
    results = json.load(open(f'results/{fn}.json'))
    for top_n in [1, 5, 10]:
        acc = round(get_acc(results, top_n)*100, 2)
        res_dict[fn].append(acc)

plt.grid(True)
plt.plot(res_dict[f'hog'], label=f'hog', marker='o', linewidth=4, markersize=8)
plt.plot(res_dict[f'statistical'], label=f'statistical', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.plot(res_dict[f'haralick'], label=f'haralick', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.legend(loc='upper left')
plt.ylim([0, 53])
plt.xticks([0,1,2], ['1', '5', '10'])
plt.title(f'Accuracy of vector-based methods')
plt.xlabel('Top N images')
plt.ylabel('Accuracy (%)')
plt.show()
# %%

    # 'gc_hist': meth.grayscale_histogram,
    # 'rgb_hist': meth.RGB_histogram,
    # 'hsv_hist': meth.HSV_histogram,
    # 'magnitude-direction': meth.gradient_magnitude_and_direction,
    # 'lbp': meth.lbp_distance,
# %%

results_filenames = [
    f'gc_hist', f'rgb_hist', f'hsv_hist', f'magnitude-direction', f'lbp'
]

res_dict = {k:[] for k in results_filenames}
for fn in results_filenames:
    results = json.load(open(f'results/{fn}.json'))
    for top_n in [1, 5, 10]:
        acc = round(get_acc(results, top_n)*100, 2)
        res_dict[fn].append(acc)

plt.grid(True)
plt.plot(res_dict[f'gc_hist'], label=f'gc_hist', marker='o', linewidth=4, markersize=8)
plt.plot(res_dict[f'rgb_hist'], label=f'rgb_hist', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.plot(res_dict[f'hsv_hist'], label=f'hsv_hist', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.plot(res_dict[f'magnitude-direction'], label=f'magnitude-direction', marker='o', linestyle='-.', linewidth=4, markersize=8)
plt.plot(res_dict[f'lbp'], label=f'lbp', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.legend(loc='upper left')
plt.ylim([0, 53])
plt.xticks([0,1,2], ['1', '5', '10'])
plt.title(f'Accuracy of histogram-based methods')
plt.xlabel('Top N images')
plt.ylabel('Accuracy (%)')
plt.show()
# %%
# %% 1/MSE i statistical - średnia arytmetyczna, harmoniczna, geometryczna

results1 = json.load(open(f'results/magnitude-direction.json'))
results2 = json.load(open(f'results/hsv_hist.json'))
results3 = json.load(open(f'results/rgb_hist.json'))
# %%
import numpy as np
import statistics as s
# %%
aggregated_1 = {}
aggregated_2 = {}
for (q_path1, data1), (q_path2, data2), (q_path3, data3) in zip(results1.items(), results2.items(), results3.items()):
    aggregated_1[q_path1] = []
    aggregated_2[q_path1] = []
    for d1, d2, d3 in zip(data1, data2, data3):
        aggregated_1[q_path1].append([d1[0], d1[1], d1[2], d2[3]*d1[3]])
        aggregated_2[q_path1].append([d1[0], d1[1], d1[2], d2[3]*d1[3]*d3[3]])

# %%
round(get_acc(aggregated, 1)*100, 2), round(get_acc(aggregated, 5)*100, 2), round(get_acc(aggregated, top_n)*100, 10)
# %%

results_filenames = [
    f'hsv_hist', f'magnitude-direction', 
]

res_dict = {k:[] for k in results_filenames}
for fn in results_filenames:
    results = json.load(open(f'results/{fn}.json'))
    for top_n in [1, 5, 10]:
        acc = round(get_acc(results, top_n)*100, 2)
        res_dict[fn].append(acc)
# %%
        
for i, agg in enumerate([aggregated_1, aggregated_2]):
    fn = f'agg_{i+1}'
    res_dict[fn] = []
    for top_n in [1, 5, 10]:
        acc = round(get_acc(agg, top_n)*100, 2)
        res_dict[fn].append(acc)
# %%
res_dict
# %%
plt.grid(True)
plt.plot(res_dict[f'agg_1'], label=f'hsv_md_hist', marker='o', linewidth=4, markersize=8)
plt.plot(res_dict[f'agg_2'], label=f'hsv_rgb_md_hist', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.plot(res_dict[f'hsv_hist'], label=f'hsv_hist', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.plot(res_dict[f'magnitude-direction'], label=f'magnitude-direction', marker='o', linestyle='-', linewidth=4, markersize=8)
plt.legend(loc='upper left')
plt.ylim([0, 53])
plt.xticks([0,1,2], ['1', '5', '10'])
plt.title(f'Accuracy of custom methods')
plt.xlabel('Top N images')
plt.ylabel('Accuracy (%)')
plt.show()
# %%
