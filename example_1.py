from two_dimension_dataset import TwoDimensionDataset
from helpers import generate_metadata_file, load_model
from cnn_model import CnnModel
from ml_model import MlModel
from engine import Engine
from performance_evaluation import Metric

# Construct dataset
dataset_name = 'hand_signs_dataset'
params = { 'batch_size': 10 }

# Training set
train_dir = 'train_signs'
generate_metadata_file(f'{dataset_name}/{train_dir}')
train_dataset = TwoDimensionDataset(f'{dataset_name}/{train_dir}')
train_dataloaders, test_dataloaders = train_dataset.getDataLoaders(params)
#print(train_dataloaders)

# Validation set
val_dir = 'test_signs'
generate_metadata_file(f'{dataset_name}/{val_dir}')
val_dataset = TwoDimensionDataset(f'{dataset_name}/{val_dir}', nFold=1)
val_dataloader = val_dataset.getDataLoader(params)


for fold in range(train_dataset.nFold):
    print('Fold {}'.format(fold))

    # ML model
    ml_model = MlModel(CnnModel(f'{dataset_name}_{fold}'))

    # Engine
    params = {
        'epochs': 30,
        'storage_dir': 'models'
        }
    engine = Engine(ml_model=ml_model, params=params)

    engine.run_training(train_dataloaders[fold], test_dataloaders[fold])

    # Validation
    best_model = CnnModel(f'{dataset_name}')
    model_name = ml_model.manifest().get_name()
    storage_dir = params['storage_dir']
    load_model(model_name, storage_dir, best_model)

    pe = engine.evaluate(val_dataloader, best_model)
    metrics = pe.get_performance_metrics()
    accuracy = metrics[Metric.ACCURACY]

    print(f'===> Fold: {fold}, Accuracy: {accuracy:.2f}')

