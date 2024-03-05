from training import train, TrainingConfig

if __name__ == "__main__":
    config = TrainingConfig(dataset="openassistant")
    train(config)