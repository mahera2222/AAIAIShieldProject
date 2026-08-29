import torch
from models.efficientnet_detector import EfficientNetTamper
from utils.preprocess import get_dataloaders
from utils.train_utils import train_detector, evaluate_model
from utils.visualize import plot_loss_curve, save_sample_predictions

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n🔥 Using device: {DEVICE}")

print("🚀 Starting AAIAIShieldProject...")

# 1) Data
train_loader, test_loader, classes = get_dataloaders(batch_size=32)

# 2) Model
model = EfficientNetTamper(num_classes=2).to(DEVICE)

# 3) Train
history = train_detector(model, train_loader, test_loader, DEVICE, epochs=12)
plot_loss_curve(history)

# Save model
torch.save(model.state_dict(), "saved_models/best_model.pth")
print("Saved model → saved_models/best_model.pth")

# 4) Evaluate
evaluate_model(model, test_loader, DEVICE)

# 5) Save prediction samples
save_sample_predictions(model, test_loader, DEVICE)
print("🎉 Pipeline completed successfully!")
