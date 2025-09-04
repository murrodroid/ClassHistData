from config import config
from train import train

if __name__ == '__main__':
    results = train(verbose=True,save_csv=True,display_figures=False,config=config)