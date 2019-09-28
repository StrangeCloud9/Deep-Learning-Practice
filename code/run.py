import data
import mlpmodel
import routine
import config
import numpy as np

# To run, make sure you have a data/ folder containing MNIST and a model/ folder along with your code/ folder

def run():
    train_loader = data.get_loader("train")
    val_loader = data.get_loader("val")
    test_loader = data.get_loader("test")
    
    if config.load_model:
        model = mlpmodel.MLP.load()
    else:
        model = mlpmodel.MLP(config.input_size, config.hidden_size, config.output_size, config.CUDA)

    routine.train(model, train_loader, val_loader)
    test_outputs = routine.predict(model, test_loader)
    try:
    	np.savetxt('prediction.csv', test_outputs, delimiter=",")
    except:
        print("prediction not saved")


if __name__ == "__main__":
    run()
