import tonic
import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose(
    [
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=3000),
    ]
)

testset = tonic.datasets.NMNIST(save_to="./data", train=False, transform=transform)

from torch.utils.data import DataLoader

testloader = DataLoader(
    testset,
    batch_size=10,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
)

frames, targets = next(iter(testloader))