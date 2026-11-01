from torchvision.datasets import CIFAR100
from torch.utils.data import Subset

dataset = CIFAR100(root="./data", train=False, download=True) # use only test images for now

print("Dataset loaded successfully!")
print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:", dataset.classes)

image, label = dataset[0]
print("\nFirst image:")
print("Image size:", image.size)
print("Label:", label)
print("Class name:", dataset.classes[label])


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
print("Number of images:", len(subset))
print("Selected classes:", selected_classes)
