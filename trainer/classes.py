from loguru import logger
import time
from collections import defaultdict
import pandas as pd


class EarlyStopping:
    def __init__(self, args, ea_modules, patience=3, delta=0, warmup=5):
        self.patience = patience
        self.ea_modules = ea_modules
        self.saved_ea_modules = [None]*len(ea_modules)
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.warmup = warmup
        self.args = args

    def __call__(self, auroc, epoch):
        if self.args.skip_ea:
            return

        if epoch < self.warmup:
            logger.info(f"Warmup epoch {epoch}, current auroc {auroc}")
            return
        score = auroc

        logger.info(
            f"best_auroc: {self.best_score}, current auroc: {score}")
        if self.best_score is None:
            self.update(score)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}, best_auroc: {self.best_score}, current auroc: {score}")
            if self.counter >= self.patience:
                self.restore()
                self.early_stop = True
        else:
            self.update(score)
            self.counter = 0

    def update(self, score):
        self.best_score = score
        for i in range(len(self.ea_modules)):
            self.saved_ea_modules[i] = self.ea_modules[i].state_dict()
            logger.info(f"EA saves {str(self.ea_modules[i].__class__)}")

    def restore(self):
        for i in range(len(self.ea_modules)):
            self.ea_modules[i].load_state_dict(self.saved_ea_modules[i])
            logger.info(f"EA restores {str(self.ea_modules[i].__class__)}")


class ElapsedTimer:
    def __init__(self):
        self.start_time = time.time()
        self.init()

    def init(self):
        self.time_dict = defaultdict(lambda: 0)
        self.len_dict = defaultdict(lambda: 0)

    def elapsed(self, name, batch_size=1):
        ret = time.time() - self.start_time
        self.reset()
        self.time_dict[name] += ret
        self.len_dict[name] += batch_size

    def reset(self):
        self.start_time = time.time()

    def stat(self):
        for k, v in self.time_dict.items():  # 평균을 구함
            self.time_dict[k] = v / self.len_dict[k]

        # dict to dataframe
        df = pd.DataFrame(list(self.time_dict.items()),
                          columns=['name', 'time'])
        total = df['time'].sum()
        df['percentage'] = df['time'] / total * 100

        self.init()
        return df
