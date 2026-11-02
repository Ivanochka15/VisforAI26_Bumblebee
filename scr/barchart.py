import csv
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename):
    with open(filename, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def calculate_accuracies(results):
    class_stats = {}
    for row in results:
        actual = row["actual_class"]
        predicted = row["predicted_class"]
        if actual not in class_stats:
            class_stats[actual] = {"correct": 0, "total": 0}
        class_stats[actual]["total"] += 1
        if actual == predicted:
            class_stats[actual]["correct"] += 1
    accuracies = {
        class_name: stats["correct"] / stats["total"]
        for class_name, stats in class_stats.items()
    }
    return accuracies


baseline_results = load_results("baseline_results.csv")
wording_results = load_results("wording_changes_results.csv")
baseline_accuracies = calculate_accuracies(baseline_results)
wording_accuracies = calculate_accuracies(wording_results)


baseline_correct = sum(
    row["actual_class"] == row["predicted_class"]
    for row in baseline_results
)

wording_correct = sum(
    row["actual_class"] == row["predicted_class"]
    for row in wording_results
)

baseline_overall = baseline_correct / len(baseline_results) * 100
wording_overall = wording_correct / len(wording_results) * 100

print(f"Baseline overall accuracy: {baseline_overall:.2f}%")
print(f"Wording changes overall accuracy: {wording_overall:.2f}%")
print(f"Difference: {wording_overall - baseline_overall:+.2f} percentage points")

### stacked bar chart ###

classes = sorted(
    baseline_accuracies.keys(),
    key=lambda class_name: baseline_accuracies[class_name]
)
baseline_values = [
    baseline_accuracies[class_name] * 100
    for class_name in classes
]
wording_values = [
    wording_accuracies[class_name] * 100
    for class_name in classes
]


# one horrizontal bar per class
fig, ax = plt.subplots(figsize=(12, 24))

y = np.arange(len(classes))

for i, (baseline, wording) in enumerate(
    zip(baseline_values, wording_values)
):
    if baseline >= wording:
        ax.barh(i, wording, color="orange", label="Wording changes" if i == 0 else "")
        ax.barh(
            i,
            baseline - wording,
            left=wording,
            color="green",
            label="Baseline" if i == 0 else ""
        )
    else:
        ax.barh(i, baseline, color="green", label="Baseline" if i == 0 else "")
        ax.barh(
            i,
            wording - baseline,
            left=baseline,
            color="orange",
            label="Wording changes" if i == 0 else ""
        )

    ax.text(
        max(baseline, wording) + 0.5,
        i,
        f"B: {baseline:.0f}% | W: {wording:.0f}%",
        va="center",
        fontsize=6
    )

ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=7)
ax.set_xlim(0, 115)
ax.set_xlabel("Accuracy (%)")
ax.set_ylabel("CIFAR-100 Class")
#ax.set_title("Per-Class Accuracy: Baseline vs. Wording Changes")

ax.legend()
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()
