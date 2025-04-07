import torch
import torch.optim as optim
from model import FloorPlanNet
from utils.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Initialize model
    model = FloorPlanNet().to(device)
    
    # Loss functions
    criterion_room = nn.CrossEntropyLoss()
    criterion_boundary = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Data loaders
    train_loader, val_loader = get_data_loaders('../dataset/jp', batch_size=8)
    
    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            room_pred, boundary_pred = model(images)
            
            # Compute losses (modify according to your label structure)
            loss_room = criterion_room(room_pred, labels[:, 0])
            loss_boundary = criterion_boundary(boundary_pred, labels[:, 1])
            total_loss = loss_room + loss_boundary
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                room_pred, boundary_pred = model(images)
                loss_room = criterion_room(room_pred, labels[:, 0])
                loss_boundary = criterion_boundary(boundary_pred, labels[:, 1])
                total_loss = loss_room + loss_boundary
                
                val_loss += total_loss.item()
        
        print(f"Validation Loss: {val_loss/len(val_loader)}")
    
    # Save model
    torch.save(model.state_dict(), 'floorplan_model.pth')

if __name__ == '__main__':
    train_model()