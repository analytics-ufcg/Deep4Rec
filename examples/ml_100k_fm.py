from deep4rec import datasets
from deep4rec import models

# Dataset
ds = datasets.build_dataset("ml-100k")

# Model
model = models.FM(ds, num_units=8, l2_regularizer=0.1)

model.train(ds, epochs=4, loss_function="rmse", optimizer="adam")
