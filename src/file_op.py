from lightai.core import *
from fastprogress import progress_bar

def create_csv(n_csv):
    inp_path = Path('inputs/train_simplified/')
    fold_path = Path('inputs/k_fold')
    fold_path.mkdir(exist_ok=True)
    for y, file_name in enumerate(inp_path.iterdir()):
        data = pd.read_csv(file_name)
        data = data.drop(['key_id', 'word'], axis=1)
        data['y'] = y
        data['cv'] = np.random.randint(0,n_csv,data.shape[0])
        for k in range(n_csv):
            file_path = fold_path/f'{k}.csv'
            chunk = data[data['cv'] == k]
            chunk = chunk.drop(['cv'], axis=1)
            if y == 0:
                chunk.to_csv(file_path, index=False)
            else:
                chunk.to_csv(file_path, mode='a', header=False, index=False)
    randomify(n_csv)


def randomify(n_csv):
    for k in progress_bar(range(n_csv)):
        fold_path = Path('inputs/k_fold')
        file_name = fold_path/f'{k}.csv'
        data = pd.read_csv(file_name)
        data = data.sample(frac=1)
        data.to_csv(file_name, index=False)