import src.project as utils
import netron

model, _ = utils.load_project()

path = utils.cache_project_model(model)

netron.start(path)