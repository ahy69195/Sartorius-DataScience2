from dataset import CellDataset
from model import UNet
from utils import *
import albumentations as A


def main():
    # Load dataset
    path = './data'
    load_model = False

    # Hyper-parameters
    path = './data'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lr = 3e-4
    epochs = 1
    transforms = A.Compose([A.Resize(512, 512),
                            A.Normalize(mean=(0.0, 0.0, 0.0),
                                        std=(1.0, 1.0, 1.0),
                                        max_pixel_value=255.0,
                                        p=1.0),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.Rotate(limit=35, p=1.0)])

    # Load the training and testing data
    cell_data_train = CellDataset(path, train=True, transform=transforms)
    cell_data_test = CellDataset(path, train=False, transform=transforms)

    # Construct Model
    model = UNet(3, 1).to(device)

    # Train/load the model
    # Uncomment to train the model
    if load_model:
        model.load_state_dict(torch.load(f'{path}/model/model.pth'))
    else:
        print('train')
        model = training(path, model, cell_data_train, lr, epochs, batch_size=1, device=device)

    # Evaluate the model
    test_loss = evaluate(model, cell_data_test, device)
    print(f'Test Loss: {test_loss:.3f} - Test Size: {len(cell_data_test)} samples')
    show_predictions(model, cell_data_test, device)


if __name__ == "__main__":
    main()
