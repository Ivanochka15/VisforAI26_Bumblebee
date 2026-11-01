
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
from transformers import CLIPModel, CLIPProcessor
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CIFAR100(root="./data", train=False, download=True) # use only test images for now
selected_classes = dataset.classes

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# baseline prompts
prompts = [
    f"a photo of a {class_name}"
    for class_name in selected_classes
]

predictions = []
actual_labels = []
similarity_scores = []

batch_size = 32
with torch.inference_mode():
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = [
            dataset[i]
            for i in range(start, end)
        ]
        images = [image for image, label in batch]
        labels = [label for image, label in batch]

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)

        batch_predictions = outputs.logits_per_image.argmax(dim=1)
        predictions.extend(batch_predictions.cpu().tolist())
        actual_labels.extend(labels)
        similarity_scores.extend(outputs.logits_per_image.cpu().tolist())

correct = sum(
    actual == predicted
    for actual, predicted in zip(actual_labels, predictions)
)
accuracy = correct / len(actual_labels) * 100
print(f"\nOverall accuracy: {accuracy:.2f}%")

print("\nAccuracy per class:")
for class_index, class_name in enumerate(selected_classes):

    class_results = [
        (actual, predicted)
        for actual, predicted in zip(actual_labels, predictions)
        if actual == class_index
    ]

    class_correct = sum(
        actual == predicted
        for actual, predicted in class_results
    )

    class_accuracy = class_correct / len(class_results) * 100

    print(f"{class_name}: {class_accuracy:.2f}%")


with open("baseline_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "image_id",
        "actual_class",
        "predicted_class",
        *[f"score_{name}" for name in selected_classes]
    ])

    for i, (actual, predicted, scores) in enumerate(zip(actual_labels, predictions, similarity_scores)):
        writer.writerow([
            i,
            selected_classes[actual],
            selected_classes[predicted],
            *scores
        ])

print("Results saved to baseline_results.csv")
