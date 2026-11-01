import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CIFAR100(root="./data", train=False, download=True) # use only test images for now
selected_classes = [
    "apple",
    "bicycle",
    "bus",
    "bear",
    "castle"
]

selected_labels = [
    dataset.class_to_idx[class_name]
    for class_name in selected_classes
]

selected_indices = [
    i for i, (_, label) in enumerate(dataset)
    if label in selected_labels
]

subset = Subset(dataset, selected_indices)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# baseline prompts
prompts = [
    f"a photo of a {class_name}"
    for class_name in selected_classes
]

# handle all img at once
# images = []
# actual_labels = []
# for image, label in subset:
#     images.append(image)
#     actual_labels.append(label)
# inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
# with torch.inference_mode():
#     outputs = model(**inputs)
#     predictions = outputs.logits_per_image.argmax(dim=1)
# predictions = predictions.cpu().tolist()

"""500 img at once are too large to handle -> use batches."""
predictions = []
actual_labels = []

batch_size = 32
with torch.inference_mode():
    for start in range(0, len(subset), batch_size):
        batch = [
            subset[i]
            for i in range(start, min(start + batch_size, len(subset)))
        ]
        images = [image for image, label in batch]
        labels = [label for image, label in batch]

        # inputs for CLIP
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        # similarity scores
        outputs = model(**inputs)
        # prompt with the highest similarity
        batch_predictions = outputs.logits_per_image.argmax(dim=1)

        predictions.extend(batch_predictions.cpu().tolist())
        actual_labels.extend(labels)


correct = sum(
    dataset.classes[actual] == selected_classes[predicted]
    for actual, predicted in zip(actual_labels, predictions)
)
accuracy = correct / len(actual_labels) * 100
print(f"\nOverall accuracy: {accuracy:.2f}%")

print("\nAccuracy per class:")
for i, class_name in enumerate(selected_classes):

    class_results = [
        (actual, predicted)
        for actual, predicted in zip(actual_labels, predictions)
        if actual == dataset.class_to_idx[class_name]
    ]

    class_correct = sum(
        dataset.classes[actual] == selected_classes[predicted]
        for actual, predicted in class_results
    )

    class_accuracy = class_correct / len(class_results) * 100

    print(f"{class_name}: {class_accuracy:.2f}%")