import torch
import torch.optim as optim
import torch.nn as nn
from model import FloorPlanNet
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    # Initialize model
    model = FloorPlanNet().to(device)

    # Loss functions
    criterion_room = nn.CrossEntropyLoss()
    criterion_boundary = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     patience=3)

    # Data loaders
    train_loader, val_loader = get_data_loaders('../dataset/jp', batch_size=8)

    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/floorplan_{timestamp}')

    best_val_loss = float('inf')

    for epoch in range(50):  # Increased epochs
        model.train()
        running_loss = 0.0

        for images, (room_labels, boundary_labels) in train_loader:
            images = images.to(device)
            room_labels = room_labels.to(device)
            boundary_labels = boundary_labels.to(device)

            # Forward pass
            room_pred, boundary_pred = model(images)

            # Compute losses
            loss_room = criterion_room(room_pred, room_labels)
            loss_boundary = criterion_boundary(boundary_pred, boundary_labels)
            total_loss = loss_room + loss_boundary

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, (room_labels, boundary_labels) in val_loader:
                images = images.to(device)
                room_labels = room_labels.to(device)
                boundary_labels = boundary_labels.to(device)

                room_pred, boundary_pred = model(images)
                loss_room = criterion_room(room_pred, room_labels)
                loss_boundary = criterion_boundary(boundary_pred,
                                                   boundary_labels)
                total_loss = loss_room + loss_boundary

                val_loss += total_loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_floorplan_model.pth')

    writer.close()
    print("Training complete")


if __name__ == '__main__':
    train_model()
