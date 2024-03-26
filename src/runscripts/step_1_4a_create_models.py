from nafld.models.model_saver_loader.model_saver import ModelLoader
from nafld.table.tables.static_table import StaticTable
from nafld.utils.initialize_environment import initialize_environment

if __name__ == "__main__":
    CONF = initialize_environment()

    base_features_table = StaticTable(name="base_features", path_to_table=CONF.DATA_BASE)

    model_loader = ModelLoader(CONF.DATA_MODELS_DIRECTORY)
