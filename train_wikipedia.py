from .training.train import train, TrainingConfiguration

if __name__ == "__main__":
    config = TrainingConfiguration()
    train(config)