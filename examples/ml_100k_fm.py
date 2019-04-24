from deep4rec import datasets
from deep4rec import models

# Dataset
ds = datasets.build_dataset("ml-100k")

# Model
model = models.FM(ds)

model.train(ds, epochs=1000, loss_function="rmse", optimizer="adam")
