import csv
import numpy as np

with open("wording_changes_results.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    results = list(reader)

classes = sorted(set(row["actual_class"] for row in results))

class_accuracies = {}
for class_name in classes:
    class_results = [
        row for row in results
        if row["actual_class"] == class_name
    ]

    correct = sum(
        row["actual_class"] == row["predicted_class"]
        for row in class_results
    )

    class_accuracies[class_name] = correct / len(class_results) * 100

sorted_classes = sorted(
    class_accuracies.items(),
    key=lambda item: item[1]
)

bottom_10 = sorted_classes[:10][::-1]
top_10 = sorted_classes[-10:][::-1]

print("\nTOP 10 CLASSES")
for name, accuracy in top_10:
    print(f"{name}: {accuracy:.2f}%")
print("\nBOTTOM 10 CLASSES")
for name, accuracy in bottom_10:
    print(f"{name}: {accuracy:.2f}%")



confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
class_to_idx = {name: index for index, name in enumerate(classes)}
for row in results:
    actual_idx = class_to_idx[row["actual_class"]]
    predicted_idx = class_to_idx[row["predicted_class"]]

    confusion_matrix[actual_idx, predicted_idx] += 1

row_sums = confusion_matrix.sum(axis=1, keepdims=True)
normalized_matrix = np.divide(
    confusion_matrix,
    row_sums,
    out=np.zeros_like(confusion_matrix, dtype=float),
    where=row_sums != 0
) * 100



print("\nMOST COMMON CONFUSIONS")

for group_name, selected in [
    ("TOP 10", top_10),
    ("BOTTOM 10", bottom_10)
]:

    print(f"\n{group_name}:")

    for class_name, accuracy in selected:

        actual_idx = class_to_idx[class_name]
        row = confusion_matrix[actual_idx].copy()

        # Exclude correct predictions
        row[actual_idx] = 0
        # Find the three most frequent incorrect predictions
        most_confused = np.argsort(row)[::-1][:3]

        print(f"\n{class_name} ({accuracy:.1f}% accuracy):")

        for idx in most_confused:
            if row[idx] > 0:
                print(
                    f"  -> {classes[idx]}: "
                    f"{row[idx]} images"
                )

